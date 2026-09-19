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

SSQ_PARTS = 256  # >= producer programs (hidden 2560 / smallest BLOCK_N 16)


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K,
    nw_ptr, ssq_in_ptr, n_parts_in, eps,
    res_ptr, ssq_out_ptr,
    NORM_IN: tl.constexpr, GLU: tl.constexpr, RES_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PARTS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    dt = w_ptr.dtype.element_ty

    if NORM_IN:
        offs_p = tl.arange(0, PARTS)
        ss = tl.load(ssq_in_ptr + offs_p[:, None] * BLOCK_M + offs_m[None, :],
                     mask=offs_p[:, None] < n_parts_in, other=0.0)
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
        tl.store(ssq_out_ptr + pid * BLOCK_M + offs_m, tl.sum(hf * hf, axis=1))
    else:
        tl.store(out_ptr + offs_mn, acc.to(dt), mask=m_mask[:, None])


class GemvConfig:
    __slots__ = ("bn", "bk", "stages", "warps")

    def __init__(self, bn, bk, stages, warps):
        self.bn, self.bk, self.stages, self.warps = bn, bk, stages, warps

    def __repr__(self):
        return f"{self.bn}x{self.bk}s{self.stages}w{self.warps}"


# Tried per projection during warmup; the fastest one that compiles wins.
GEMV_CANDIDATES = (
    GemvConfig(16, 256, 3, 4),
    GemvConfig(16, 512, 3, 4),
    GemvConfig(32, 256, 3, 4),
    GemvConfig(32, 128, 4, 4),
    GemvConfig(64, 128, 4, 4),
    GemvConfig(16, 128, 5, 4),
    GemvConfig(64, 256, 3, 8),
)


def gemv(x, w, out, m, n, k, cfg, block_m, ssq_in, ssq_out, eps,
         norm_w=None, n_parts=1, res=None, glu=False):
    """``x[:m] @ w.T`` into ``out`` (or into ``res`` in place when ``res`` is
    given), with the fusions described above. ``ssq_in``/``ssq_out`` are
    ``[SSQ_PARTS, block_m]`` FP32 buffers and are always passed, even when
    unused, so every call site compiles to the same signature."""
    norm_in = norm_w is not None
    res_out = res is not None
    _gemv_kernel[(n // cfg.bn,)](
        x, w, res if res_out else out, m, n, k,
        norm_w if norm_in else w, ssq_in, n_parts, eps,
        res if res_out else out, ssq_out,
        NORM_IN=norm_in, GLU=glu, RES_OUT=res_out,
        BLOCK_M=block_m, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk, PARTS=SSQ_PARTS,
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
    k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr,
    stride_cb, stride_ch, stride_cs, chunk, num_splits, sm_scale_log2, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
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
    _attn_fused_split_kernel[(attn.batch, attn.nkv, attn.splits)](
        qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, attn.po, attn.pm, attn.pl,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        attn.chunk, attn.splits, attn.sm_scale_log2, eps,
        NQ=attn.nq, NKV=attn.nkv, GROUP=attn.group, D=attn.d,
        BLOCK_H=max(16, triton.next_power_of_2(attn.group)), BLOCK_N=attn.BLOCK_N,
        num_warps=4,
    )
    _decode_attn_reduce_kernel[(attn.batch, attn.nq)](
        attn.po, attn.pm, attn.pl, out, attn.splits,
        NQ=attn.nq, D=attn.d, SPLITS=max(2, triton.next_power_of_2(attn.splits)),
        num_warps=4,
    )
    return out
