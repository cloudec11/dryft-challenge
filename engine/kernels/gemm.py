"""The projection kernel: the whole decode step is five launches of this.

At decode every projection has ``M = batch`` rows against thousands of output
columns, so none of them is a matmul in any useful sense -- each is one weight
matrix streamed past a handful of rows.  The step reads 8.05 GB of weights
whatever we do, so the only questions are how close to HBM speed each read
runs and how many launches the step pays for.  This kernel answers both:

* **One kernel, five roles.**  The elementwise work that would otherwise sit
  between projections rides along as a prologue or an epilogue, so a layer is
  five launches (QKV, attention, O, gate/up, down) rather than ten:

  ``NORM``  RMSNorm of x in the prologue, from the partial sums of squares
            the producing kernel left behind.
  ``GLU``   a second weight block (``up``) streamed beside the first
            (``gate``), with SwiGLU in the epilogue.
  ``RES``   the residual add, done in place, plus this program's partial sum
            of squares for the next RMSNorm.

* **Split-K without a second launch.**  Splitting K is what lets a wide
  column tile pay for itself: a wide tile divides the activation re-reads but
  empties the device, and the split fills it back up without adding any x
  traffic.  Every engine before this one paid a reduce launch for that, which
  at ~2.8 us and 36 layers is 0.1 ms of a 4.8 ms step for each role that
  splits -- enough to eat the win.  Here the last program to finish a column
  tile folds the slices itself (``FIXUP``), in the CUTLASS "last block does
  the reduction" pattern: each program stores its FP32 partial, syncs its
  block, and bumps an atomic counter whose release ordering publishes the
  store; the program that reads ``SPLIT - 1`` back knows every slice is
  visible, sums them in slice order -- deterministically -- and runs the
  epilogue.  Nothing spins, so nothing can deadlock on a block that was never
  scheduled.

  The counter is reset by that same program, so a captured graph can replay
  the kernel any number of times.  If the pattern ever fails its load-time
  check the engine falls back to ``_proj_reduce_kernel``, which does the same
  arithmetic in a second launch.

Numerics: BF16 rounding happens exactly where Transformers 4.51.3 rounds it
(``reference.py`` holds the twins these are checked against).  Sums are
reordered -- tile sums, FP32 partials, slices added before the single round
to BF16 -- which the contract allows.  No cast boundary moves.
"""

import torch
import triton
import triton.language as tl

# The RMSNorm prologue reads its producer's partial sums of squares in blocks
# of this many at a time.  A block, not the whole array: the partial count is
# ``N / BLOCK_N`` of the producing role, up to 160, and a [160, BLOCK_M] tile
# held in registers is a spill at any useful BLOCK_M.  Sixteen at a time is
# 8 registers per thread and the loop runs a handful of times.
PART_CHUNK = 16


@triton.jit
def _epilogue(
    acc, acc_u, y_ptr, res_ptr, ssq_out_ptr, offs_m, offs_n, m_mask, part_id, N,
    GLU: tl.constexpr, RES: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """Shared by the unsplit, fixup and reduce paths so they cannot drift."""
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if GLU:
        dt = y_ptr.dtype.element_ty
        # Reference MLP: bf16(bf16(silu(bf16(gate))) * bf16(up)).
        g = acc.to(dt).to(tl.float32)
        u = acc_u.to(dt).to(tl.float32)
        a = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
        tl.store(y_ptr + off, (a * u).to(dt), mask=m_mask[:, None])
    elif RES:
        dt = res_ptr.dtype.element_ty
        d = acc.to(dt).to(tl.float32)
        r = tl.load(res_ptr + off, mask=m_mask[:, None], other=0.0).to(tl.float32)
        # Reference: hidden = residual + hidden, materialised in BF16.
        h = (r + d).to(dt)
        tl.store(res_ptr + off, h, mask=m_mask[:, None])
        hf = tl.where(m_mask[:, None], h.to(tl.float32), 0.0)
        tl.store(ssq_out_ptr + part_id * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1),
                 mask=m_mask)
    else:
        dt = y_ptr.dtype.element_ty
        tl.store(y_ptr + off, acc.to(dt), mask=m_mask[:, None])


@triton.jit
def _rstd_from_parts(ssq_in_ptr, offs_m, m_mask, n_parts, K, eps,
                     BLOCK_M: tl.constexpr, PART_STEPS: tl.constexpr,
                     SS_STRIDE: tl.constexpr):
    """Finish the RMSNorm the previous kernel started.

    The producer left one partial sum of squares per column tile; the sum is
    over the whole row, so the partials are added before the K slice is taken.
    Every role that normalises has ``K = hidden``, which is the dimension
    Qwen3RMSNorm reduces over.

    The trip count is a constexpr, so the loop unrolls and the loads issue
    together.  It was a runtime loop once, to keep the kernel to a single
    specialisation -- but this runs in the prologue of every program of three
    of the five roles, ahead of the weight stream it is holding up, and the
    producer's partial count takes one of three values in a whole run.
    """
    offs_p = tl.arange(0, PART_CHUNK)
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for step in tl.static_range(PART_STEPS):
        offs = step * PART_CHUNK + offs_p
        ss = tl.load(
            ssq_in_ptr + offs[:, None] * SS_STRIDE + offs_m[None, :],
            mask=(offs[:, None] < n_parts) & m_mask[None, :], other=0.0,
        )
        total += tl.sum(ss, axis=0)
    return tl.math.rsqrt(total / K + eps)


@triton.jit
def _proj_kernel(
    x_ptr, w_ptr, y_ptr, res_ptr, part_ptr, ctr_ptr, ssq_in_ptr, ssq_out_ptr,
    nw_ptr, M, N, K, part_s_stride, n_parts, eps,
    NORM: tl.constexpr, GLU: tl.constexpr, RES: tl.constexpr,
    SPLIT: tl.constexpr, FIXUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PART_STEPS: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """``y = x[:M] @ w.T`` for one tile of output columns and one slice of K.

    Grid is ``(N // BLOCK_N, ceil(M / BLOCK_M), SPLIT)``.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty
    k_span = K // SPLIT
    k_start = s * k_span

    if NORM:
        rstd = _rstd_from_parts(ssq_in_ptr, offs_m, m_mask, n_parts, K, eps,
                                BLOCK_M=BLOCK_M, PART_STEPS=PART_STEPS,
                                SS_STRIDE=SS_STRIDE)

    x_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * K + k_start + offs_k[None, :]
    # A [BLOCK_K, BLOCK_N] window on W[N, K] (K contiguous), so acc += x @ W.T
    # and every load runs down the contiguous axis.
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * K + k_start + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GLU:
        wu_ptrs = w_ptrs + N.to(tl.int64) * K  # the up rows follow the gate rows
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(k_start, k_start + k_span, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        if NORM:
            # Qwen3RMSNorm: normalise in FP32, round to BF16, then weight.
            nw = tl.load(nw_ptr + k0 + offs_k).to(tl.float32)
            xn = (x.to(tl.float32) * rstd[:, None]).to(dt)
            x = (nw[None, :] * xn.to(tl.float32)).to(dt)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        if GLU:
            wu = tl.load(wu_ptrs)
            acc_u += tl.dot(x, wu)
            wu_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    tile = pid_m * (N // BLOCK_N) + pid_n
    if SPLIT == 1:
        _epilogue(acc, acc_u if GLU else acc, y_ptr, res_ptr, ssq_out_ptr,
                  offs_m, offs_n, m_mask, pid_n, N,
                  GLU=GLU, RES=RES, SS_STRIDE=SS_STRIDE)
    else:
        dst = (part_ptr + s.to(tl.int64) * part_s_stride
               + offs_m[:, None].to(tl.int64) * N + offs_n[None, :])
        tl.store(dst, acc, mask=m_mask[:, None], cache_modifier=".cg")
        if GLU:
            tl.store(dst + SPLIT * part_s_stride, acc_u, mask=m_mask[:, None],
                     cache_modifier=".cg")
        if FIXUP:
            # Publish this slice, then claim a ticket.  The barrier orders
            # every thread's store ahead of the atomic, and the atomic's
            # release makes them visible to whoever acquires the counter.
            tl.debug_barrier()
            ticket = tl.atomic_add(ctr_ptr + tile, 1, sem="acq_rel", scope="gpu")
            if ticket == SPLIT - 1:
                tl.store(ctr_ptr + tile, 0)  # ready for the next replay
                src = (part_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :])
                total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for j in tl.static_range(SPLIT):
                    total += tl.load(src + j * part_s_stride, mask=m_mask[:, None],
                                     other=0.0, cache_modifier=".cg")
                if GLU:
                    total_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    for j in tl.static_range(SPLIT):
                        total_u += tl.load(
                            src + (SPLIT + j) * part_s_stride, mask=m_mask[:, None],
                            other=0.0, cache_modifier=".cg")
                else:
                    total_u = total
                _epilogue(total, total_u, y_ptr, res_ptr, ssq_out_ptr,
                          offs_m, offs_n, m_mask, pid_n, N,
                          GLU=GLU, RES=RES, SS_STRIDE=SS_STRIDE)


@triton.jit
def _proj_reduce_kernel(
    part_ptr, y_ptr, res_ptr, ssq_out_ptr, M, N, part_s_stride,
    GLU: tl.constexpr, RES: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """The fallback reduction, in its own launch.

    Same arithmetic and the same epilogue as the in-kernel fixup: the slices
    add in FP32 and round to BF16 once.  Used when the fixup path fails its
    load-time check, and as the thing the fixup path is raced against.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * part_s_stride + off, mask=m_mask[:, None],
                       other=0.0)
    if GLU:
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            acc_u += tl.load(part_ptr + (SPLIT + s) * part_s_stride + off,
                             mask=m_mask[:, None], other=0.0)
    else:
        acc_u = acc
    _epilogue(acc, acc_u, y_ptr, res_ptr, ssq_out_ptr, offs_m, offs_n, m_mask,
              pid_n, N, GLU=GLU, RES=RES, SS_STRIDE=SS_STRIDE)


@triton.jit
def _proj_vec_kernel(
    x_ptr, w_ptr, y_ptr, res_ptr, ssq_in_ptr, ssq_out_ptr, nw_ptr,
    N, K, n_parts, eps,
    NORM: tl.constexpr, GLU: tl.constexpr, RES: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PART_STEPS: tl.constexpr,
    SS_STRIDE: tl.constexpr,
):
    """Batch-1 specialisation: FP32 FMA into a ``[BLOCK_N, BLOCK_K]``
    accumulator, reduced once at the end.

    At one row the tile kernel pads x to 16 rows for ``tl.dot`` and pays the
    tensor-core layout conversion on every iteration for a single useful row.
    This loop is nothing but wide weight loads.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty

    if NORM:
        offs_p = tl.arange(0, PART_CHUNK)
        total = 0.0
        for step in tl.static_range(PART_STEPS):
            offs = step * PART_CHUNK + offs_p
            ss = tl.load(ssq_in_ptr + offs * SS_STRIDE, mask=offs < n_parts,
                         other=0.0)
            total += tl.sum(ss, axis=0)
        rstd = tl.math.rsqrt(total / K + eps)

    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    if GLU:
        wu_ptrs = w_ptrs + N.to(tl.int64) * K
        acc_u = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k0 + offs_k)
        if NORM:
            nw = tl.load(nw_ptr + k0 + offs_k).to(tl.float32)
            xn = (x.to(tl.float32) * rstd).to(dt)
            x = (nw * xn.to(tl.float32)).to(dt)
        xf = x.to(tl.float32)
        w = tl.load(w_ptrs)
        acc += w.to(tl.float32) * xf[None, :]
        if GLU:
            wu = tl.load(wu_ptrs)
            acc_u += wu.to(tl.float32) * xf[None, :]
            wu_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    y = tl.sum(acc, axis=1)
    if GLU:
        g = y.to(dt).to(tl.float32)
        u = tl.sum(acc_u, axis=1).to(dt).to(tl.float32)
        a = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
        tl.store(y_ptr + offs_n, (a * u).to(dt))
    elif RES:
        r = tl.load(res_ptr + offs_n).to(tl.float32)
        h = (r + y.to(dt).to(tl.float32)).to(dt)
        tl.store(res_ptr + offs_n, h)
        hf = h.to(tl.float32)
        tl.store(ssq_out_ptr + pid * SS_STRIDE, tl.sum(hf * hf, axis=0))
    else:
        tl.store(y_ptr + offs_n, y.to(dt))


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------

def n_tiles(tile, n):
    """Column tiles, which is how many partial sums of squares a producing
    role hands to the next RMSNorm."""
    return -(-n // tile.bn)


def buffers(block_m, widest_n, max_split, max_tiles, device):
    """The scratch a split-K plan needs: FP32 partials and tile counters.

    ``widest_n`` is the widest role allowed to split, counting both weight
    blocks of a GLU role, so one buffer serves every projection.
    """
    part = torch.empty(max_split * block_m * widest_n, dtype=torch.float32,
                       device=device)
    ctr = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    return part, ctr


def project(x, w, y, m, n, k, tile, ssq_in, ssq_out, eps, *, norm_w=None,
            glu=False, res=None, n_parts=1, ss_stride=1, part=None, ctr=None):
    """Launch one projection.

    ``x``        ``[m, k]`` BF16, contiguous.  ``w`` is ``[n, k]`` BF16
                 (``[2 * n, k]`` for a GLU role), ``y`` is ``[m, n]`` BF16.
    ``res``      ``[m, hidden]`` BF16 updated in place when set; ``y`` is then
                 unused and the epilogue writes ``ssq_out``.
    ``norm_w``   RMSNorm gain applied in the prologue when set; the row's sum
                 of squares is read from ``ssq_in[:n_parts]``.
    ``tile``     a :class:`planner.Tile`.  ``part`` and ``ctr`` must be given
                 when it splits K.

    Everything is caller-owned; nothing here allocates, so the call is safe to
    capture in a CUDA graph.
    """
    norm = norm_w is not None
    if n % tile.bn or k % tile.bk or k % tile.split:
        raise ValueError(f"tile {tile} does not divide n={n} k={k}")
    part_steps = -(-n_parts // PART_CHUNK) if norm else 1
    if tile.kind == "vec":
        _proj_vec_kernel[(n // tile.bn,)](
            x, w, y, res, ssq_in, ssq_out, norm_w, n, k, n_parts, eps,
            NORM=norm, GLU=glu, RES=res is not None,
            BLOCK_N=tile.bn, BLOCK_K=tile.bk, PART_STEPS=part_steps,
            SS_STRIDE=ss_stride, num_warps=tile.warps, num_stages=tile.stages,
        )
        return

    tiles_m = -(-m // tile.bm)
    tiles_n = n // tile.bn
    part_s_stride = tile.bm * tiles_m * n
    _proj_kernel[(tiles_n, tiles_m, tile.split)](
        x, w, y, res, part, ctr, ssq_in, ssq_out, norm_w,
        m, n, k, part_s_stride, n_parts, eps,
        NORM=norm, GLU=glu, RES=res is not None,
        SPLIT=tile.split, FIXUP=tile.fixup,
        BLOCK_M=tile.bm, BLOCK_N=tile.bn, BLOCK_K=tile.bk,
        PART_STEPS=part_steps, SS_STRIDE=ss_stride,
        num_warps=tile.warps, num_stages=tile.stages,
    )
    if tile.split > 1 and not tile.fixup:
        _proj_reduce_kernel[(tiles_n, tiles_m)](
            part, y, res, ssq_out, m, n, part_s_stride,
            GLU=glu, RES=res is not None, SPLIT=tile.split,
            BLOCK_M=tile.bm, BLOCK_N=tile.bn, SS_STRIDE=ss_stride,
            num_warps=4, num_stages=1,
        )
