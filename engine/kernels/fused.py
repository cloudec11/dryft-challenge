"""Fused single-token decode kernels.

Decode runs every projection at M = batch rows, which is pure weight
streaming, and v1 spent roughly as much time in the small elementwise
launches between projections as in the extra GEMM work. These kernels read
each weight once and fold those elementwise ops into the GEMM itself:

* RES_OUT epilogue: ``h = bf16(h + bf16(x @ W^T))`` written back into the
  residual buffer in place (each program owns its columns), plus each
  program's partial sum of ``h^2`` per row into ``ssq_out[pid]``.
* NORM_IN prologue: the consumer adds those partials up (a reordered FP32
  reduction), forms rstd, and normalises h tile by tile exactly as
  Qwen3RMSNorm rounds: ``bf16(w * bf16(h * rstd))``.
* GLU epilogue: gate and up come from the same program, each rounded to BF16,
  then ``bf16(bf16(silu(gate)) * up)``, as the reference MLP does.
* Attention: the Q/K head norm, RoPE and the KV-cache write happen inside the
  split-K attention kernel.

Per layer that is 6 launches (QKV, attention split, attention reduce, O,
gate/up, down) instead of 10. BF16 rounding points are the reference's
(see ``ops.py``); only reduction order differs.
"""

import torch
import triton
import triton.language as tl

from kernels.ops import _decode_attn_reduce_kernel

SSQ_PARTS = 1024  # rows of the sum-of-squares buffers: >= producer programs (2560 / BLOCK_N 4)


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K,
    nw_ptr, ssq_in_ptr, n_parts_in, eps,
    res_ptr, ssq_out_ptr,
    NORM_IN: tl.constexpr, GLU: tl.constexpr, RES_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PARTS: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    # Tensor-core tile version: x is padded to BLOCK_M >= 16 rows for tl.dot.
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty

    if NORM_IN:
        offs_p = tl.arange(0, PARTS)
        ss = tl.load(ssq_in_ptr + offs_p[:, None] * SS_STRIDE + offs_m[None, :],
                     mask=(offs_p[:, None] < n_parts_in) & m_mask[None, :], other=0.0)
        rstd = tl.math.rsqrt(tl.sum(ss, axis=0) / K + eps)

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    # [BLOCK_K, BLOCK_N] view of W[N, K] (K contiguous), so acc += x @ W^T.
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GLU:
        wu_ptrs = w_ptrs + N.to(tl.int64) * K  # up rows follow the gate rows
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        if NORM_IN:
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

    offs_mn = offs_m[:, None] * N + offs_n[None, :]
    if GLU:
        g = acc.to(dt).to(tl.float32)
        u = acc_u.to(dt).to(tl.float32)
        a = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
        tl.store(out_ptr + offs_mn, (a * u).to(dt), mask=m_mask[:, None])
    elif RES_OUT:
        d = acc.to(dt).to(tl.float32)
        r = tl.load(res_ptr + offs_mn, mask=m_mask[:, None], other=0.0).to(tl.float32)
        h = (r + d).to(dt)
        tl.store(res_ptr + offs_mn, h, mask=m_mask[:, None])
        hf = tl.where(m_mask[:, None], h.to(tl.float32), 0.0)
        tl.store(ssq_out_ptr + pid * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1), mask=m_mask)
    else:
        tl.store(out_ptr + offs_mn, acc.to(dt), mask=m_mask[:, None])


@triton.jit
def _gemv_vec_kernel(
    x_ptr, w_ptr, out_ptr, N, K,
    nw_ptr, ssq_in_ptr, n_parts_in, eps,
    res_ptr, ssq_out_ptr,
    NORM_IN: tl.constexpr, GLU: tl.constexpr, RES_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PARTS: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    # Single-row version (batch 1): plain FMA into a [BLOCK_N, BLOCK_K] FP32
    # accumulator, reduced once at the end. No padding to 16 rows, no
    # tensor-core layout conversions; the loop is just wide weight loads.
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty

    if NORM_IN:
        offs_p = tl.arange(0, PARTS)
        ss = tl.load(ssq_in_ptr + offs_p * SS_STRIDE, mask=offs_p < n_parts_in, other=0.0)
        rstd = tl.math.rsqrt(tl.sum(ss, axis=0) / K + eps)

    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    if GLU:
        wu_ptrs = w_ptrs + N.to(tl.int64) * K
        acc_u = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k0 + offs_k)
        if NORM_IN:
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
        tl.store(out_ptr + offs_n, (a * u).to(dt))
    elif RES_OUT:
        r = tl.load(res_ptr + offs_n).to(tl.float32)
        h = (r + y.to(dt).to(tl.float32)).to(dt)
        tl.store(res_ptr + offs_n, h)
        hf = h.to(tl.float32)
        tl.store(ssq_out_ptr + pid * SS_STRIDE, tl.sum(hf * hf, axis=0))
    else:
        tl.store(out_ptr + offs_n, y.to(dt))


MAX_SPLIT = 8  # rows of the FP32 partial buffer a split-K projection needs


@triton.jit
def _gemv_splitk_kernel(
    x_ptr, w_ptr, part_ptr, M, N, K,
    SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (column tile, K slice). N = 2560 tiles into only 160
    # programs at BLOCK_N = 16, so a plain grid leaves the second wave a
    # fifth full on 132 SMs; splitting K multiplies the program count and
    # lets BLOCK_N grow, which also cuts the repeated reads of x.
    pid_n = tl.program_id(0)
    s = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_span = K // SPLIT
    k_start = s * k_span
    x_ptrs = x_ptr + offs_m[:, None] * K + k_start + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * K + k_start + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_span, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    dst = part_ptr + s * (BLOCK_M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(dst, acc, mask=m_mask[:, None])


@triton.jit
def _gemv_splitk_reduce_kernel(
    part_ptr, res_ptr, ssq_out_ptr, M, N,
    SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    # Sums the FP32 slices and runs the RES_OUT epilogue. The slices add up
    # in FP32 and round to BF16 once, exactly as the unsplit GEMM does.
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    off = offs_m[:, None] * N + offs_n[None, :]
    dt = res_ptr.dtype.element_ty
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * (BLOCK_M * N) + off, mask=m_mask[:, None], other=0.0)
    d = acc.to(dt).to(tl.float32)
    r = tl.load(res_ptr + off, mask=m_mask[:, None], other=0.0).to(tl.float32)
    h = (r + d).to(dt)
    tl.store(res_ptr + off, h, mask=m_mask[:, None])
    hf = tl.where(m_mask[:, None], h.to(tl.float32), 0.0)
    tl.store(ssq_out_ptr + pid * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1), mask=m_mask)


class GemvConfig:
    __slots__ = ("bn", "bk", "stages", "warps", "vec", "split")

    def __init__(self, bn, bk, stages, warps, vec=False, split=1):
        self.bn, self.bk, self.stages, self.warps = bn, bk, stages, warps
        self.vec, self.split = vec, split

    def __repr__(self):
        kind = "vec" if self.vec else ("k%d-" % self.split if self.split > 1 else "mma")
        return f"{kind}{self.bn}x{self.bk}s{self.stages}w{self.warps}"


# Tried per projection during warmup; the fastest one that compiles wins.
# Batch 1 tries the single-row kernel first, then the tile kernel.
GEMV_VEC_CANDIDATES = (
    GemvConfig(8, 512, 2, 4, vec=True),
    GemvConfig(4, 512, 2, 4, vec=True),
    GemvConfig(16, 256, 2, 4, vec=True),
    GemvConfig(16, 512, 2, 8, vec=True),
)
GEMV_CANDIDATES = (
    GemvConfig(16, 256, 3, 4),
    GemvConfig(16, 512, 3, 4),
    GemvConfig(32, 256, 3, 4),
    GemvConfig(32, 128, 4, 4),
    GemvConfig(64, 128, 4, 4),
    GemvConfig(64, 256, 3, 8),
    # Narrow K, for large BLOCK_M. A hidden workload at batch 32-128 runs
    # these projections with BLOCK_M 32-128, where every tile above keeps
    # more than the 227 KiB an SM has and gets filtered out unbuilt; the
    # public shapes (batch 1, 4, 16) never exercise that. Last in the list,
    # so they still have to beat the incumbent by MARGIN.
    GemvConfig(32, 64, 4, 4),
    GemvConfig(64, 64, 4, 4),
)


# Only the residual-writing projections (N = hidden) use split-K: the others
# already tile into enough programs, and their prologues/epilogues would have
# to move into the reduce kernel.
GEMV_SPLITK_CANDIDATES = (
    GemvConfig(64, 256, 3, 4, split=2),
    GemvConfig(32, 256, 3, 4, split=2),
    GemvConfig(64, 128, 4, 4, split=4),
    GemvConfig(32, 128, 4, 4, split=4),
    GemvConfig(128, 128, 3, 8, split=2),
    GemvConfig(64, 128, 4, 4, split=8),
    GemvConfig(32, 128, 4, 4, split=8),
)


def candidates(m, split_ok=False):
    base = (GEMV_VEC_CANDIDATES + GEMV_CANDIDATES[:3]) if m == 1 else GEMV_CANDIDATES
    if not split_ok:
        return base
    # Split-K first: it is the candidate that fixes the wave quantization of
    # the N = 2560 projections, and the budget may not reach the whole list.
    return GEMV_SPLITK_CANDIDATES + base[:3]


def gemv(x, w, out, m, n, k, cfg, block_m, ssq_in, ssq_out, eps,
         norm_w=None, n_parts=1, res=None, glu=False, parts_block=SSQ_PARTS, part=None):
    """``x[:m] @ w.T`` into ``out`` (or into ``res`` in place when ``res`` is
    given), with the fusions described above. ``ssq_in``/``ssq_out`` are
    ``[SSQ_PARTS, block_m]`` FP32 buffers and are always passed, even when
    unused, so every call site compiles to the same signature.
    ``parts_block`` is a power of two >= ``n_parts``."""
    norm_in = norm_w is not None
    res_out = res is not None
    dst = res if res_out else out
    nw = norm_w if norm_in else w
    if cfg.split > 1:
        assert res_out and not norm_in and not glu
        _gemv_splitk_kernel[(n // cfg.bn, cfg.split)](
            x, w, part, m, n, k,
            SPLIT=cfg.split, BLOCK_M=block_m, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk,
            num_warps=cfg.warps, num_stages=cfg.stages,
        )
        _gemv_splitk_reduce_kernel[(n // cfg.bn,)](
            part, res, ssq_out, m, n,
            SPLIT=cfg.split, BLOCK_M=block_m, BLOCK_N=cfg.bn, SS_STRIDE=block_m,
            num_warps=4,
        )
        return
    if cfg.vec:
        assert m == 1
        _gemv_vec_kernel[(n // cfg.bn,)](
            x, w, dst, n, k, nw, ssq_in, n_parts, eps, dst, ssq_out,
            NORM_IN=norm_in, GLU=glu, RES_OUT=res_out,
            BLOCK_N=cfg.bn, BLOCK_K=cfg.bk, PARTS=parts_block, SS_STRIDE=block_m,
            num_warps=cfg.warps, num_stages=cfg.stages,
        )
        return
    _gemv_kernel[(n // cfg.bn,)](
        x, w, dst, m, n, k, nw, ssq_in, n_parts, eps, dst, ssq_out,
        NORM_IN=norm_in, GLU=glu, RES_OUT=res_out,
        BLOCK_M=block_m, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk,
        PARTS=parts_block, SS_STRIDE=block_m,
        num_warps=cfg.warps, num_stages=cfg.stages,
    )


@triton.jit
def _embed_kernel(ids_ptr, emb_ptr, h_ptr, ssq_ptr, H, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    tok = tl.load(ids_ptr + m).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(emb_ptr + tok * H + cols, mask=mask, other=0.0)
    tl.store(h_ptr + m.to(tl.int64) * H + cols, x, mask=mask)
    xf = x.to(tl.float32)
    tl.store(ssq_ptr + m, tl.sum(xf * xf, axis=0))  # partial 0 of row m


def embed_ssq(ids, emb, h, ssq):
    """``h[m] = emb[ids[m]]`` and ``ssq[0, m] = sum(h[m]^2)`` (one partial)."""
    m = ids.shape[0]
    hidden = emb.shape[1]
    _embed_kernel[(m,)](ids, emb, h, ssq, hidden, BLOCK=triton.next_power_of_2(hidden), num_warps=4)


@triton.jit
def _attn_fused_split_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, out_ptr,
    stride_cb, stride_ch, stride_cs, chunk, num_splits, sm_scale_log2, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, ONE_SPLIT: tl.constexpr,
):
    # ops._decode_attn_split_kernel with ops._qkv_post_kernel folded in. Every
    # program rebuilds its group's rotated q and the new k/v from the raw QKV
    # row (a few hundred bytes), substitutes the new k/v for cache slot
    # ``pos`` (not yet written), and the one program whose split owns ``pos``
    # stores them. Splits cover disjoint ranges, so no program reads a slot
    # another program of this launch writes.
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    p = tl.load(pos_ptr)
    seq_len = p + 1
    start = split * chunk
    end = tl.minimum(start + chunk, seq_len)
    dt = k_ptr.dtype.element_ty

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    partner = (offs_d + D // 2) % D
    first_half = offs_d < D // 2
    h_mask = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    row = qkv_ptr + b.to(tl.int64) * ((NQ + 2 * NKV) * D)
    c = tl.load(cos_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)
    s = tl.load(sin_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)

    # q: head RMSNorm then RoPE, rounded as Qwen3RMSNorm / apply_rotary_pos_emb.
    xq = tl.load(row + heads[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0).to(tl.float32)
    xqp = tl.load(row + heads[:, None] * D + partner[None, :], mask=h_mask[:, None], other=0.0).to(tl.float32)
    rq = tl.math.rsqrt(tl.sum(xq * xq, axis=1) / D + eps)
    wq = tl.load(qw_ptr + offs_d).to(tl.float32)
    wqp = tl.load(qw_ptr + partner).to(tl.float32)
    yq = (wq[None, :] * (xq * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yqp = (wqp[None, :] * (xqp * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    rotq = tl.where(first_half[None, :], -yqp, yqp)
    q = ((yq * c[None, :]).to(dt).to(tl.float32) + (rotq * s[None, :]).to(dt).to(tl.float32)).to(dt)

    # This step's k (normed, rotated) and v (raw) for kv-head kvh.
    kcol = row + (NQ + kvh) * D
    xk = tl.load(kcol + offs_d).to(tl.float32)
    xkp = tl.load(kcol + partner).to(tl.float32)
    rk = tl.math.rsqrt(tl.sum(xk * xk, axis=0) / D + eps)
    wk = tl.load(kw_ptr + offs_d).to(tl.float32)
    wkp = tl.load(kw_ptr + partner).to(tl.float32)
    yk = (wk * (xk * rk).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    ykp = (wkp * (xkp * rk).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    rotk = tl.where(first_half, -ykp, ykp)
    k_new = ((yk * c).to(dt).to(tl.float32) + (rotk * s).to(dt).to(tl.float32)).to(dt)
    v_new = tl.load(row + (NQ + NKV + kvh) * D + offs_d)

    base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        is_new = (offs_n == p)[:, None]
        kv_off = base + offs_n[:, None].to(tl.int64) * stride_cs + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        k = tl.where(is_new, k_new[None, :], k)
        sc = tl.dot(q, tl.trans(k)) * sm_scale_log2
        sc = tl.where(n_mask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        pr = tl.math.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, axis=1)
        v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        v = tl.where(is_new, v_new[None, :], v)
        acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
        m_i = m_new

    owner = (start <= p) & (p < start + chunk)
    new_off = base + p.to(tl.int64) * stride_cs + offs_d
    tl.store(k_ptr + new_off, k_new, mask=(offs_d < D) & owner)
    tl.store(v_ptr + new_off, v_new, mask=(offs_d < D) & owner)

    if ONE_SPLIT:
        # One split spans the sequence: normalise here and skip the reduce.
        o = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :], o,
                 mask=h_mask[:, None])
    else:
        part = (b * NQ + heads).to(tl.int64) * num_splits + split
        tl.store(po_ptr + part[:, None] * D + offs_d[None, :], acc, mask=h_mask[:, None])
        tl.store(pm_ptr + part, m_i, mask=h_mask)
        tl.store(pl_ptr + part, l_i, mask=h_mask)


def attention_fused(attn, qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, eps):
    """Norm + RoPE + cache write + attention for one token per sequence.

    ``attn`` is an ``ops.DecodeAttention`` (split layout and partial buffers);
    ``qkv`` is the raw ``[B, (nq + 2 nkv) * D]`` projection. Returns
    ``[B, nq * D]`` like ``DecodeAttention.__call__``."""
    out = torch.empty((attn.batch, attn.nq * attn.d), dtype=qkv.dtype, device=qkv.device)
    one = attn.splits == 1
    _attn_fused_split_kernel[(attn.batch, attn.nkv, attn.splits)](
        qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache,
        attn.po, attn.pm, attn.pl, out,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        attn.chunk, attn.splits, attn.sm_scale_log2, eps,
        NQ=attn.nq, NKV=attn.nkv, GROUP=attn.group, D=attn.d,
        BLOCK_H=max(16, triton.next_power_of_2(attn.group)), BLOCK_N=attn.block_n,
        ONE_SPLIT=one, num_warps=attn.num_warps, num_stages=attn.num_stages,
    )
    if not one:
        _decode_attn_reduce_kernel[(attn.batch, attn.nq)](
            attn.po, attn.pm, attn.pl, out, attn.splits,
            NQ=attn.nq, D=attn.d, SPLITS=max(2, triton.next_power_of_2(attn.splits)),
            num_warps=4,
        )
    return out
