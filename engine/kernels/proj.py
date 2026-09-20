"""Skinny-GEMM projection kernels: the whole decode step lives in these.

Decode runs every projection at ``M = batch`` rows, so none of them is a
matmul in any useful sense -- each is a stream of one weight matrix past a
handful of rows. The step reads 8.05 GB of weights whatever we do, so the
only questions are how close to HBM speed each read runs, and how many
launches the step pays for.

Both are answered here:

* One kernel covers all five roles. The elementwise work that used to sit
  between projections rides along as a prologue or an epilogue, so a layer is
  five launches (QKV, attention, O, gate/up, down) instead of ten.

  - ``NORM``  RMSNorm of x in the prologue, from the partial sums of squares
              the producing kernel left in ``ssq_in``.
  - ``GLU``   a second weight block (``up``) streamed beside the first
              (``gate``), with SwiGLU in the epilogue.
  - ``RES``   the residual add done in place on the residual buffer, plus
              this program's partial sum of squares for the next RMSNorm.

* Tiles are chosen by measurement, from a space built around the device's SM
  count rather than a fixed list. What limits a streaming kernel here is
  wave quantisation -- ``N/BLOCK_N`` programs spread over 132 SMs, where a
  1.2-wave grid wastes 40% of the second wave -- and how many bytes each SM
  keeps in flight. :func:`candidates` enumerates the space, scores every
  point on both, and hands the tuner an ordered shortlist.

Numerics: BF16 rounding happens exactly where Transformers 4.51.3 rounds
(``ref.py`` holds the twins these are checked against). Sums are reordered,
which the contract allows; no formula and no cast boundary moves.
"""

import math

import torch
import triton
import triton.language as tl

# Rows of the sum-of-squares hand-off buffers. A producer writes one partial
# per program; the producing roles (O, down) have N = hidden = 2560, so the
# narrowest tile offered (BLOCK_N 4, batch 1 only) asks for 640 of them.
SSQ_PARTS = 1024
# PARTS is a constexpr, so a role that does not normalise would otherwise
# compile a second specialisation for every value the plan happens to hold.
# It reads nothing from the buffer, so pin it.
NO_PARTS = 2
# K slices a split-K projection may use. Only the N = hidden roles ever split
# (see :func:`candidates`), so the FP32 partial buffer stays at
# MAX_SPLIT * BLOCK_M * hidden floats.
MAX_SPLIT = 16
# H100 shared memory per SM, and the most a single block may ask for before
# we stop bothering to compile it.
SMEM_PER_SM = 227 * 1024
SMEM_MAX_BLOCK = 200 * 1024


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------


@triton.jit
def _proj_epilogue(
    acc, acc_u, y_ptr, res_ptr, ssq_out_ptr, offs_m, offs_n, m_mask, pid_n, N,
    GLU: tl.constexpr, RES: tl.constexpr, BLOCK_N: tl.constexpr,
    SS_STRIDE: tl.constexpr,
):
    """Shared by the fused and the split-K path, so the two cannot drift."""
    off = offs_m[:, None] * N + offs_n[None, :]
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
        tl.store(ssq_out_ptr + pid_n * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1),
                 mask=m_mask)
    else:
        dt = y_ptr.dtype.element_ty
        tl.store(y_ptr + off, acc.to(dt), mask=m_mask[:, None])


@triton.jit
def _proj_kernel(
    x_ptr, w_ptr, y_ptr, res_ptr, part_ptr, ssq_in_ptr, ssq_out_ptr, nw_ptr,
    M, N, K, n_parts, eps,
    NORM: tl.constexpr, GLU: tl.constexpr, RES: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PARTS: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """``y = x[:M] @ w.T`` for one tile of output columns and one slice of K.

    Grid is ``(N // BLOCK_N, SPLIT)``. With ``SPLIT == 1`` the program runs
    the epilogue itself; with more, it writes an FP32 partial and
    :func:`_proj_reduce_kernel` runs the epilogue once the slices are in.
    """
    pid_n = tl.program_id(0)
    s = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty
    k_span = K // SPLIT
    k_start = s * k_span

    if NORM:
        # rstd is over the whole row, so the partials are summed before the K
        # slice is taken. Every role that normalises has K = hidden, which is
        # the dimension Qwen3RMSNorm reduces over.
        offs_p = tl.arange(0, PARTS)
        ss = tl.load(
            ssq_in_ptr + offs_p[:, None] * SS_STRIDE + offs_m[None, :],
            mask=(offs_p[:, None] < n_parts) & m_mask[None, :], other=0.0,
        )
        rstd = tl.math.rsqrt(tl.sum(ss, axis=0) / K + eps)

    x_ptrs = x_ptr + offs_m[:, None] * K + k_start + offs_k[None, :]
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

    if SPLIT > 1:
        dst = (part_ptr + s.to(tl.int64) * (BLOCK_M * N)
               + offs_m[:, None] * N + offs_n[None, :])
        tl.store(dst, acc, mask=m_mask[:, None])
    else:
        _proj_epilogue(
            acc, acc_u if GLU else acc, y_ptr, res_ptr, ssq_out_ptr,
            offs_m, offs_n, m_mask, pid_n, N,
            GLU=GLU, RES=RES, BLOCK_N=BLOCK_N, SS_STRIDE=SS_STRIDE,
        )


@triton.jit
def _proj_reduce_kernel(
    part_ptr, y_ptr, res_ptr, ssq_out_ptr, M, N,
    GLU: tl.constexpr, RES: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """Sum the FP32 slices, then run the same epilogue the fused path runs.

    The slices add in FP32 and round to BF16 once, so this is a reordering of
    one sum rather than a different function. The grid is ``N // BLOCK_N``,
    the same as the unsplit kernel's, so how many sums of squares a consumer
    reads does not depend on whether K was split.
    """
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * (BLOCK_M * N) + off, mask=m_mask[:, None], other=0.0)
    _proj_epilogue(
        acc, acc, y_ptr, res_ptr, ssq_out_ptr, offs_m, offs_n, m_mask, pid_n, N,
        GLU=GLU, RES=RES, BLOCK_N=BLOCK_N, SS_STRIDE=SS_STRIDE,
    )


@triton.jit
def _proj_vec_kernel(
    x_ptr, w_ptr, y_ptr, res_ptr, ssq_in_ptr, ssq_out_ptr, nw_ptr,
    N, K, n_parts, eps,
    NORM: tl.constexpr, GLU: tl.constexpr, RES: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PARTS: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    """Batch-1 specialisation: plain FP32 FMA into a ``[BLOCK_N, BLOCK_K]``
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
        offs_p = tl.arange(0, PARTS)
        ss = tl.load(ssq_in_ptr + offs_p * SS_STRIDE, mask=offs_p < n_parts, other=0.0)
        rstd = tl.math.rsqrt(tl.sum(ss, axis=0) / K + eps)

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
# Tile configurations
# --------------------------------------------------------------------------


class Config:
    __slots__ = ("bn", "bk", "split", "stages", "warps", "vec", "ctas", "waves", "flight")

    def __init__(self, bn, bk, split=1, stages=3, warps=4, vec=False):
        self.bn, self.bk, self.split = bn, bk, split
        self.stages, self.warps, self.vec = stages, warps, vec
        self.ctas = 0
        self.waves = 1.0
        self.flight = 0

    def __repr__(self):
        kind = "vec" if self.vec else ("k%d-" % self.split if self.split > 1 else "")
        return f"{kind}{self.bn}x{self.bk}s{self.stages}w{self.warps}"


def smem_bytes(cfg, block_m, glu):
    """Shared memory one block needs: the x tile plus one (or two, for
    gate/up) weight tiles, per pipeline stage."""
    if cfg.vec:
        return cfg.stages * (2 if glu else 1) * cfg.bn * cfg.bk * 2
    per_stage = block_m * cfg.bk + (2 if glu else 1) * cfg.bk * cfg.bn
    return cfg.stages * per_stage * 2


def _blocks_per_sm(cfg, block_m, glu):
    """How many of these blocks an SM can hold, by shared memory and by
    threads. Used only for ranking; the tuner measures the truth."""
    smem = smem_bytes(cfg, block_m, glu)
    by_smem = max(1, SMEM_PER_SM // max(smem, 1))
    by_threads = max(1, 2048 // (32 * cfg.warps))
    return min(by_smem, by_threads, 32)


def _score(cfg, block_m, glu, n, k, sm_count, prefer_wide=False):
    """Rank a candidate on the two things that decide a weight stream.

    ``waves``  programs divided by SMs. A grid of 160 programs on 132 SMs
               runs as two rounds for 1.21 rounds of work, so 40% of the
               second round is idle -- which is most of the gap between the
               measured 2.0 TB/s and the 3.35 TB/s the device can do.
    ``flight`` bytes of weight an SM keeps outstanding. Too few and the
               kernel is latency-bound however well it is balanced.
    """
    cfg.ctas = (n // cfg.bn) * cfg.split
    rounds = cfg.ctas / sm_count
    cfg.waves = rounds / math.ceil(rounds) if rounds > 0 else 0.0
    per_stage_weight = (2 if glu else 1) * cfg.bk * cfg.bn * 2
    cfg.flight = _blocks_per_sm(cfg, block_m, glu) * cfg.stages * per_stage_weight
    # Efficiency first, in coarse buckets so a 1% modelling difference cannot
    # outrank a real bandwidth difference; then bytes in flight. A wide tile
    # is preferred for the roles that produce sums of squares, because its
    # program count is what every consumer's prologue then has to read.
    return (-round(cfg.waves, 2), -min(cfg.flight, 128 * 1024),
            -cfg.bn if prefer_wide else cfg.bn)


# Tried first for every role, and the incumbent a measured candidate has to
# beat by the tuner's margin: the tile v1-v10 ran with.
_SAFE = (16, 256, 3, 4)
_BN = (16, 32, 64, 128)
_BK = (128, 256, 512, 1024)
_STAGES = (3, 4)
_WARPS = (4, 8)


def candidates(m, n, k, glu, sm_count, limit=11, prefer_wide=False):
    """An ordered shortlist of tiles to race for one projection.

    The first entry is the incumbent; everything after it is sorted by
    predicted wave efficiency and then by bytes in flight. Split-K is offered
    only where the unsplit grid is too small to fill the device -- the
    ``N = hidden`` projections, whose 2560 columns tile into 160 programs.

    ``prefer_wide`` breaks ties towards a wider tile, for the roles whose
    program count becomes every consumer's prologue cost.
    """
    block_m = max(16, triton.next_power_of_2(m)) if m > 1 else 16
    pool = []
    if m == 1:
        pool += [Config(bn, bk, vec=True, stages=st, warps=wp)
                 for bn in (4, 8, 16) for bk in (256, 512)
                 for st in (2, 3) for wp in (4, 8)]
    splits = (1,)
    if n // 16 < 2 * sm_count:
        splits = tuple(s for s in (1, 2, 4, 8) if s <= MAX_SPLIT)
    for bn in _BN:
        for bk in _BK:
            for sp in splits:
                for st in _STAGES:
                    for wp in _WARPS:
                        pool.append(Config(bn, bk, split=sp, stages=st, warps=wp))

    out = []
    seen = set()
    for cfg in pool:
        if n % cfg.bn or cfg.bn > n or k % (cfg.bk * cfg.split):
            continue
        if cfg.split > 1 and glu:
            continue  # the FP32 partial buffer is sized for one output block
        if not cfg.vec and smem_bytes(cfg, block_m, glu) > SMEM_MAX_BLOCK:
            continue
        key = repr(cfg)
        if key in seen:
            continue
        seen.add(key)
        _score(cfg, block_m, glu, n, k, sm_count, prefer_wide)
        out.append(cfg)
    out.sort(key=lambda c: _score(c, block_m, glu, n, k, sm_count, prefer_wide))

    head = []
    bn, bk, st, wp = _SAFE
    if m > 1 and n % bn == 0 and k % bk == 0:
        safe = Config(bn, bk, stages=st, warps=wp)
        _score(safe, block_m, glu, n, k, sm_count, prefer_wide)
        head.append(safe)
    for cfg in out:
        if len(head) >= limit:
            break
        if head and repr(cfg) == repr(head[0]):
            continue
        head.append(cfg)
    return head


# --------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------


def project(x, w, y, m, n, k, cfg, block_m, ssq_in, ssq_out, eps,
            norm_w=None, n_parts=1, res=None, glu=False, parts_block=SSQ_PARTS,
            part=None):
    """``x[:m] @ w.T``, with the fusions this module exists for.

    ``y``        destination, or ``None`` when ``res`` takes the result
    ``res``      residual buffer updated in place (the ``RES`` epilogue)
    ``norm_w``   RMSNorm weight; its presence selects the ``NORM`` prologue
    ``ssq_in``   ``[SSQ_PARTS, block_m]`` FP32 partials the prologue sums
    ``ssq_out``  where a ``RES`` epilogue leaves its own partials
    ``n_parts``  how many rows of ``ssq_in`` are live
    ``part``     ``[MAX_SPLIT, block_m, n]`` FP32 scratch for split-K

    Every argument is passed on every call, even where a specialisation does
    not read it, so each call site compiles exactly one kernel signature.
    """
    norm = norm_w is not None
    res_out = res is not None
    dst = res if res_out else y
    nw = norm_w if norm else w
    if not norm:
        # Nothing is read from the buffer, so pin the constexpr rather than
        # compiling a second specialisation per plan.
        parts_block, n_parts = NO_PARTS, 1
    elif n_parts > parts_block:
        raise ValueError(f"{n_parts} partials do not fit in {parts_block}")

    if cfg.vec:
        _proj_vec_kernel[(n // cfg.bn,)](
            x, w, dst, dst, ssq_in, ssq_out, nw,
            n, k, n_parts, eps,
            NORM=norm, GLU=glu, RES=res_out,
            BLOCK_N=cfg.bn, BLOCK_K=cfg.bk, PARTS=parts_block, SS_STRIDE=block_m,
            num_warps=cfg.warps, num_stages=cfg.stages,
        )
        return

    _proj_kernel[(n // cfg.bn, cfg.split)](
        x, w, dst, dst, part, ssq_in, ssq_out, nw,
        m, n, k, n_parts, eps,
        NORM=norm, GLU=glu, RES=res_out, SPLIT=cfg.split,
        BLOCK_M=block_m, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk,
        PARTS=parts_block, SS_STRIDE=block_m,
        num_warps=cfg.warps, num_stages=cfg.stages,
    )
    if cfg.split > 1:
        _proj_reduce_kernel[(n // cfg.bn,)](
            part, dst, dst, ssq_out, m, n,
            GLU=glu, RES=res_out, SPLIT=cfg.split,
            BLOCK_M=block_m, BLOCK_N=cfg.bn, SS_STRIDE=block_m,
            num_warps=4,
        )


def n_tiles(cfg, n):
    """Programs the epilogue runs in, and so how many sums of squares a
    consumer of this projection has to read. Split-K reduces in a grid of the
    same width, so this does not depend on ``cfg.split``."""
    return n // cfg.bn


def part_buffer(block_m, hidden, device):
    """FP32 scratch for split-K. Only the ``N = hidden`` roles split, so this
    is 2.6 MB at batch 16 rather than a slice per MLP column."""
    return torch.empty((MAX_SPLIT, block_m, hidden), dtype=torch.float32, device=device)
