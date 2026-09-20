"""Elementwise kernels for prefill and for the reference decode path.

The fused decode step folds all of this into the projection kernels, so these
only run where that is not possible: the prefill, which is compute-bound and
happily uses cuBLAS for its GEMMs, and the plain decode step that the fused
one is checked and raced against.

Rounding follows Transformers 4.51.3 exactly; ``ref.py`` holds the PyTorch
twins every one of these is compared with before it is allowed to run.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Residual add + RMSNorm
# --------------------------------------------------------------------------


@triton.jit
def _add_rms_norm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, n_cols, eps,
    HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dt = out_ptr.dtype.element_ty
    h = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        h = (r + h).to(dt)  # the reference materialises residual + hidden in BF16
        tl.store(res_out_ptr + row * n_cols + cols, h, mask=mask)
        h = h.to(tl.float32)
    variance = tl.sum(h * h, axis=0) / n_cols
    normed = (h * tl.math.rsqrt(variance + eps)).to(dt)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # weight * normed.to(bf16), rounded once more on the way out.
    tl.store(out_ptr + row * n_cols + cols,
             (w.to(tl.float32) * normed.to(tl.float32)).to(dt), mask=mask)


def add_rms_norm(x, residual, weight, eps):
    """``(rms_norm(residual + x), residual + x)``; with ``residual=None``,
    ``(rms_norm(x), x)``. ``x`` is ``[M, N]`` contiguous."""
    m, n = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    if residual is None:
        _add_rms_norm_kernel[(m,)](x, x, weight, out, out, n, eps,
                                   HAS_RES=False, BLOCK=block, num_warps=8)
        return out, x
    res_out = torch.empty_like(x)
    _add_rms_norm_kernel[(m,)](x, residual, weight, out, res_out, n, eps,
                               HAS_RES=True, BLOCK=block, num_warps=8)
    return out, res_out


# --------------------------------------------------------------------------
# Q/K head RMSNorm + RoPE + KV-cache write
# --------------------------------------------------------------------------


@triton.jit
def _qkv_post_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, stride_cb, stride_ch, stride_cs, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    # One program per BLOCK_HD heads of a row, so a 2048-token prefill
    # launches a few thousand full programs instead of ~100k one-warp ones.
    row = tl.program_id(0)  # b * T + t
    hh = tl.program_id(1) * BLOCK_HD + tl.arange(0, BLOCK_HD)
    b = row // T
    t = row % T
    pos = tl.load(pos_ptr) + t
    d = tl.arange(0, D)
    partner = (d + D // 2) % D
    dt = q_out_ptr.dtype.element_ty
    is_q = hh < NQ
    is_k = (hh >= NQ) & (hh < NQ + NKV)
    is_v = (hh >= NQ + NKV) & (hh < NQ + 2 * NKV)
    h_mask = hh < NQ + 2 * NKV

    src = qkv_ptr + row.to(tl.int64) * ((NQ + 2 * NKV) * D)
    x = tl.load(src + hh[:, None] * D + d[None, :], mask=h_mask[:, None],
                other=0.0).to(tl.float32)
    xp = tl.load(src + hh[:, None] * D + partner[None, :], mask=h_mask[:, None],
                 other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    wq = tl.load(qw_ptr + d).to(tl.float32)
    wk = tl.load(kw_ptr + d).to(tl.float32)
    wqp = tl.load(qw_ptr + partner).to(tl.float32)
    wkp = tl.load(kw_ptr + partner).to(tl.float32)
    w = tl.where(is_q[:, None], wq[None, :], wk[None, :])
    wp = tl.where(is_q[:, None], wqp[None, :], wkp[None, :])
    y = (w * (x * rstd[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yp = (wp * (xp * rstd[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    # rotate_half: the first half takes -x2, the second half takes x1.
    rot = tl.where(d[None, :] < D // 2, -yp, yp)
    c = tl.load(cos_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
    s = tl.load(sin_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
    out = ((y * c[None, :]).to(dt).to(tl.float32)
           + (rot * s[None, :]).to(dt).to(tl.float32)).to(dt)

    tl.store(q_out_ptr + row.to(tl.int64) * (NQ * D) + hh[:, None] * D + d[None, :],
             out, mask=is_q[:, None])
    cache = b.to(tl.int64) * stride_cb + pos.to(tl.int64) * stride_cs + d[None, :]
    tl.store(k_cache_ptr + cache + (hh - NQ)[:, None] * stride_ch, out, mask=is_k[:, None])
    tl.store(v_cache_ptr + cache + (hh - NQ - NKV)[:, None] * stride_ch, x.to(dt),
             mask=is_v[:, None])


def qkv_post(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv, eps):
    """``qkv`` is ``[B*T, (nq + 2 nkv) * D]``, batch-major; token ``t`` sits at
    absolute position ``pos[0] + t``. Writes normalised, rotated K and raw V
    into the caches (``[B, nkv, capacity, D]``) and returns Q as
    ``[B*T, nq * D]``."""
    m = qkv.shape[0]
    d = q_weight.shape[0]
    block_hd = 16
    q_out = torch.empty((m, nq * d), dtype=qkv.dtype, device=qkv.device)
    _qkv_post_kernel[(m, triton.cdiv(nq + 2 * nkv, block_hd))](
        qkv, q_weight, k_weight, cos, sin, pos, q_out, k_cache, v_cache,
        T, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), eps,
        NQ=nq, NKV=nkv, D=d, BLOCK_HD=block_hd, num_warps=4,
    )
    return q_out


# --------------------------------------------------------------------------
# SwiGLU
# --------------------------------------------------------------------------


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, inter, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    dt = out_ptr.dtype.element_ty
    g = tl.load(gu_ptr + row * (2 * inter) + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * (2 * inter) + inter + cols, mask=mask, other=0.0).to(tl.float32)
    # Reference: act_fn(gate) materialised in BF16, then * up, rounded again.
    act = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
    tl.store(out_ptr + row * inter + cols, (act * u).to(dt), mask=mask)


def silu_mul(gate_up):
    """``gate_up`` is ``[M, 2 I]`` with the gate columns first."""
    m, two_i = gate_up.shape
    inter = two_i // 2
    out = torch.empty((m, inter), dtype=gate_up.dtype, device=gate_up.device)
    block = 1024
    _silu_mul_kernel[(m, triton.cdiv(inter, block))](gate_up, out, inter,
                                                     BLOCK=block, num_warps=4)
    return out


# --------------------------------------------------------------------------
# Embedding, with the first sum of squares for the fused step's first RMSNorm
# --------------------------------------------------------------------------


@triton.jit
def _embed_kernel(ids_ptr, emb_ptr, h_ptr, ssq_ptr, H, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    tok = tl.load(ids_ptr + m).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(emb_ptr + tok * H + cols, mask=mask, other=0.0)
    tl.store(h_ptr + m.to(tl.int64) * H + cols, x, mask=mask)
    xf = tl.where(mask, x.to(tl.float32), 0.0)
    tl.store(ssq_ptr + m, tl.sum(xf * xf, axis=0))  # partial 0 of row m


def embed(ids, emb_w, h, ssq):
    """``h[m] = emb[ids[m]]``, and ``ssq[0, m] = sum(h[m] ** 2)``."""
    m = ids.shape[0]
    hidden = emb_w.shape[1]
    _embed_kernel[(m,)](ids, emb_w, h, ssq, hidden,
                        BLOCK=triton.next_power_of_2(hidden), num_warps=4)
