"""Fused Triton kernels for the Qwen3 forward, plus plain-PyTorch twins.

Every kernel reproduces the reference's BF16 rounding points (Transformers
4.51.3, ``modeling_qwen3``). Wherever the reference materialises a BF16
tensor, the kernel casts to the output dtype at that same point; arithmetic
between those points runs in FP32, which is what PyTorch's elementwise kernels
do internally. Reductions may be reordered; formulas may not change.

The ``*_ref`` functions compute the same thing with PyTorch ops. They are the
fallback if Triton cannot compile in the sandbox and the reference in tests.
All of them are CUDA-graph safe: positions and lengths are read from device
tensors, never from Python ints that change per step.
"""

import math

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
        # Reference: hidden = residual + hidden, materialised in BF16.
        r = tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        h = (r + h).to(dt)
        tl.store(res_out_ptr + row * n_cols + cols, h, mask=mask)
        h = h.to(tl.float32)
    variance = tl.sum(h * h, axis=0) / n_cols
    normed = (h * tl.math.rsqrt(variance + eps)).to(dt)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Reference: weight * normed.to(bf16) -> rounded once more to BF16.
    y = (w.to(tl.float32) * normed.to(tl.float32)).to(dt)
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


def add_rms_norm(x, residual, weight, eps):
    """Returns ``(rms_norm(residual + x), residual + x)``; with ``residual``
    None, returns ``(rms_norm(x), x)``. ``x`` is ``[M, N]`` contiguous."""
    m, n = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    if residual is None:
        _add_rms_norm_kernel[(m,)](
            x, x, weight, out, out, n, eps,
            HAS_RES=False, BLOCK=block, num_warps=8,
        )
        return out, x
    res_out = torch.empty_like(x)
    _add_rms_norm_kernel[(m,)](
        x, residual, weight, out, res_out, n, eps,
        HAS_RES=True, BLOCK=block, num_warps=8,
    )
    return out, res_out


def add_rms_norm_ref(x, residual, weight, eps):
    h = x if residual is None else residual + x
    hf = h.to(torch.float32)
    variance = hf.pow(2).mean(-1, keepdim=True)
    normed = (hf * torch.rsqrt(variance + eps)).to(h.dtype)
    return weight * normed, h


# --------------------------------------------------------------------------
# Q/K head RMSNorm + RoPE + KV-cache write
# --------------------------------------------------------------------------


@triton.jit
def _qkv_post_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, stride_cb, stride_ch, stride_cs, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)  # b * T + t
    head = tl.program_id(1)  # [0, NQ) q, [NQ, NQ+NKV) k, then v
    b = row // T
    t = row % T
    pos = tl.load(pos_ptr) + t
    d = tl.arange(0, D)
    partner = (d + D // 2) % D
    dt = q_out_ptr.dtype.element_ty
    src = qkv_ptr + row.to(tl.int64) * ((NQ + 2 * NKV) * D) + head * D
    cache_off = b.to(tl.int64) * stride_cb + pos.to(tl.int64) * stride_cs

    if head < NQ + NKV:
        is_q = head < NQ
        x = tl.load(src + d).to(tl.float32)
        xp = tl.load(src + partner).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        wq = tl.load(qw_ptr + d).to(tl.float32)
        wk = tl.load(kw_ptr + d).to(tl.float32)
        wqp = tl.load(qw_ptr + partner).to(tl.float32)
        wkp = tl.load(kw_ptr + partner).to(tl.float32)
        w = tl.where(is_q, wq, wk)
        wp = tl.where(is_q, wqp, wkp)
        # Head RMSNorm, rounded exactly as Qwen3RMSNorm rounds.
        y = (w * (x * rstd).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        yp = (wp * (xp * rstd).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        # rotate_half: first half takes -x2, second half takes x1.
        rot = tl.where(d < D // 2, -yp, yp)
        c = tl.load(cos_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
        s = tl.load(sin_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
        # (q * cos) + (rotate_half(q) * sin), each product rounded to BF16.
        a = (y * c).to(dt).to(tl.float32)
        r = (rot * s).to(dt).to(tl.float32)
        out = (a + r).to(dt)
        if is_q:
            tl.store(q_out_ptr + row.to(tl.int64) * (NQ * D) + head * D + d, out)
        else:
            tl.store(k_cache_ptr + cache_off + (head - NQ) * stride_ch + d, out)
    else:
        v = tl.load(src + d)
        tl.store(v_cache_ptr + cache_off + (head - NQ - NKV) * stride_ch + d, v)


def qkv_post(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv, eps):
    """``qkv`` is ``[M, (nq + 2 nkv) * D]`` with ``M = B * T`` rows ordered
    batch-major. Token ``t`` sits at absolute position ``pos[0] + t``.
    Writes normalised, rotated K and raw V into ``k_cache``/``v_cache``
    (``[B, nkv, capacity, D]``) and returns Q as ``[M, nq * D]``."""
    m = qkv.shape[0]
    d = q_weight.shape[0]
    q_out = torch.empty((m, nq * d), dtype=qkv.dtype, device=qkv.device)
    _qkv_post_kernel[(m, nq + 2 * nkv)](
        qkv, q_weight, k_weight, cos, sin, pos, q_out, k_cache, v_cache,
        T, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), eps,
        NQ=nq, NKV=nkv, D=d, num_warps=1,
    )
    return q_out


@triton.jit
def _qkv_post_rows_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, stride_cb, stride_ch, stride_cs, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    # Same arithmetic as _qkv_post_kernel, but one program handles BLOCK_HD
    # heads of a row instead of one head, so a long prefill launches ~16x
    # fewer (and fuller) programs.
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
    x = tl.load(src + hh[:, None] * D + d[None, :], mask=h_mask[:, None], other=0.0).to(tl.float32)
    xp = tl.load(src + hh[:, None] * D + partner[None, :], mask=h_mask[:, None], other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    wq = tl.load(qw_ptr + d).to(tl.float32)
    wk = tl.load(kw_ptr + d).to(tl.float32)
    wqp = tl.load(qw_ptr + partner).to(tl.float32)
    wkp = tl.load(kw_ptr + partner).to(tl.float32)
    w = tl.where(is_q[:, None], wq[None, :], wk[None, :])
    wp = tl.where(is_q[:, None], wqp[None, :], wkp[None, :])
    y = (w * (x * rstd[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yp = (wp * (xp * rstd[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    rot = tl.where(d[None, :] < D // 2, -yp, yp)
    c = tl.load(cos_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
    s = tl.load(sin_ptr + pos.to(tl.int64) * D + d).to(tl.float32)
    out = ((y * c[None, :]).to(dt).to(tl.float32) + (rot * s[None, :]).to(dt).to(tl.float32)).to(dt)
    tl.store(q_out_ptr + row.to(tl.int64) * (NQ * D) + hh[:, None] * D + d[None, :], out,
             mask=is_q[:, None])
    cache = b.to(tl.int64) * stride_cb + pos.to(tl.int64) * stride_cs + d[None, :]
    tl.store(k_cache_ptr + cache + (hh - NQ)[:, None] * stride_ch, out, mask=is_k[:, None])
    tl.store(v_cache_ptr + cache + (hh - NQ - NKV)[:, None] * stride_ch, x.to(dt), mask=is_v[:, None])


def qkv_post_rows(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv, eps):
    """Drop-in replacement for :func:`qkv_post` (same inputs and outputs)."""
    m = qkv.shape[0]
    d = q_weight.shape[0]
    block_hd = 16
    q_out = torch.empty((m, nq * d), dtype=qkv.dtype, device=qkv.device)
    _qkv_post_rows_kernel[(m, triton.cdiv(nq + 2 * nkv, block_hd))](
        qkv, q_weight, k_weight, cos, sin, pos, q_out, k_cache, v_cache,
        T, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), eps,
        NQ=nq, NKV=nkv, D=d, BLOCK_HD=block_hd, num_warps=4,
    )
    return q_out


def _rms_ref(x, weight, eps):
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    return weight * (xf * torch.rsqrt(variance + eps)).to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qkv_post_ref(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv, eps):
    m = qkv.shape[0]
    d = q_weight.shape[0]
    b = m // T
    x = qkv.view(b, T, nq + 2 * nkv, d)
    q = _rms_ref(x[:, :, :nq], q_weight, eps)
    k = _rms_ref(x[:, :, nq:nq + nkv], k_weight, eps)
    v = x[:, :, nq + nkv:]
    positions = pos.to(torch.int64) + torch.arange(T, device=qkv.device)
    c = cos.index_select(0, positions)[None, :, None, :]
    s = sin.index_select(0, positions)[None, :, None, :]
    q = (q * c) + (_rotate_half(q) * s)
    k = (k * c) + (_rotate_half(k) * s)
    k_cache.index_copy_(2, positions, k.transpose(1, 2))
    v_cache.index_copy_(2, positions, v.transpose(1, 2))
    return q.reshape(m, nq * d)


# --------------------------------------------------------------------------
# SwiGLU activation: silu(gate) * up
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
    """``gate_up`` is ``[M, 2 I]`` (gate columns first); returns ``[M, I]``."""
    m, two_i = gate_up.shape
    inter = two_i // 2
    out = torch.empty((m, inter), dtype=gate_up.dtype, device=gate_up.device)
    block = 1024
    _silu_mul_kernel[(m, triton.cdiv(inter, block))](gate_up, out, inter, BLOCK=block, num_warps=4)
    return out


def silu_mul_ref(gate_up):
    inter = gate_up.shape[1] // 2
    return torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]


# --------------------------------------------------------------------------
# Single-token GQA attention over the cache (split-K "flash decoding")
# --------------------------------------------------------------------------


@triton.jit
def _decode_attn_split_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, po_ptr, pm_ptr, pl_ptr, out_ptr,
    stride_cb, stride_ch, stride_cs, chunk, num_splits, sm_scale_log2,
    NQ: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, ONE_SPLIT: tl.constexpr,
    STREAM: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    seq_len = tl.load(pos_ptr) + 1  # the current token's K/V is already written
    start = split * chunk
    end = tl.minimum(start + chunk, seq_len)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    q = tl.load(
        q_ptr + b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :],
        mask=h_mask[:, None], other=0.0,
    )
    base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        kv_off = base + offs_n[:, None].to(tl.int64) * stride_cs + offs_d[None, :]
        # The cache is read once per step and never reused, so with STREAM it
        # does not hold L2 at the expense of activations that are re-read.
        if STREAM:
            k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0,
                        eviction_policy="evict_first")
        else:
            k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale_log2
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if STREAM:
            v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0,
                        eviction_policy="evict_first")
        else:
            v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        # As in FlashAttention, probabilities enter the PV product in the
        # value dtype with FP32 accumulation.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if ONE_SPLIT:
        # The only split spans the whole sequence, so the reduce kernel would
        # just divide by l_i: do it here and skip that launch.
        o = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :], o,
                 mask=h_mask[:, None])
    else:
        part = (b * NQ + heads).to(tl.int64) * num_splits + split
        tl.store(po_ptr + part[:, None] * D + offs_d[None, :], acc, mask=h_mask[:, None])
        tl.store(pm_ptr + part, m_i, mask=h_mask)
        tl.store(pl_ptr + part, l_i, mask=h_mask)


@triton.jit
def _decode_attn_reduce_kernel(
    po_ptr, pm_ptr, pl_ptr, out_ptr, num_splits,
    NQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, D)
    s_mask = offs_s < num_splits
    part = (b * NQ + h).to(tl.int64) * num_splits + offs_s
    m = tl.load(pm_ptr + part, mask=s_mask, other=float("-inf"))
    l = tl.load(pl_ptr + part, mask=s_mask, other=0.0)
    m_max = tl.max(m, axis=0)  # finite: split 0 always holds key 0
    w = tl.math.exp2(m - m_max)  # empty splits weigh exactly zero
    l_sum = tl.sum(l * w, axis=0)
    o = tl.load(po_ptr + part[:, None] * D + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / l_sum
    dst = out_ptr + b.to(tl.int64) * (NQ * D) + h * D + offs_d
    tl.store(dst, acc.to(out_ptr.dtype.element_ty))


# (BLOCK_N, target programs, num_warps, num_stages) tried at warmup. At a
# large batch x context the KV read is the whole step, so the tile shape and
# the split count matter more than any heuristic can predict.
ATTN_CANDIDATES = (
    (64, 264, 4, 3),
    (128, 264, 4, 3),
    (64, 528, 4, 3),
    (128, 132, 8, 3),
    (64, 132, 4, 4),
    (256, 264, 8, 3),
    # target 1 forces a single split, which skips the reduce launch entirely
    # (36 fewer kernels per step); worth it once batch * kv heads fills the
    # device on its own.
    (64, 1, 4, 3),
    (128, 1, 8, 3),
)


class DecodeAttention:
    """Attention of one new token per sequence over ``[0, pos[0]]`` of a
    fixed-capacity cache. The split layout depends only on batch size and
    capacity, so one instance serves every step of a captured graph."""

    BLOCK_N = 64

    def __init__(self, batch, capacity, nq, nkv, head_dim, device, target_programs=264,
                 block_n=None, num_warps=4, num_stages=3, stream=False):
        self.nq, self.nkv, self.d = nq, nkv, head_dim
        self.group = nq // nkv
        self.block_n = block_n or self.BLOCK_N
        self.num_warps, self.num_stages = num_warps, num_stages
        self.stream = stream  # L2 policy for the KV loads, raced whole-step
        max_splits = triton.cdiv(capacity, self.block_n)
        want = max(1, triton.cdiv(target_programs, batch * nkv))
        splits = min(want, max_splits)
        self.chunk = triton.cdiv(triton.cdiv(capacity, splits), self.block_n) * self.block_n
        self.splits = triton.cdiv(capacity, self.chunk)
        self.batch = batch
        self.sm_scale_log2 = (1.0 / math.sqrt(head_dim)) * 1.4426950408889634
        self.po = torch.empty((batch * nq * self.splits, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.empty((batch * nq * self.splits,), dtype=torch.float32, device=device)
        self.pl = torch.empty((batch * nq * self.splits,), dtype=torch.float32, device=device)

    def __call__(self, q, k_cache, v_cache, pos):
        out = torch.empty((self.batch, self.nq * self.d), dtype=q.dtype, device=q.device)
        one = self.splits == 1
        _decode_attn_split_kernel[(self.batch, self.nkv, self.splits)](
            q, k_cache, v_cache, pos, self.po, self.pm, self.pl, out,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.chunk, self.splits, self.sm_scale_log2,
            NQ=self.nq, GROUP=self.group, D=self.d,
            BLOCK_H=max(16, triton.next_power_of_2(self.group)), BLOCK_N=self.block_n,
            ONE_SPLIT=one, STREAM=self.stream,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if not one:
            _decode_attn_reduce_kernel[(self.batch, self.nq)](
                self.po, self.pm, self.pl, out, self.splits,
                NQ=self.nq, D=self.d, SPLITS=max(2, triton.next_power_of_2(self.splits)),
                num_warps=4,
            )
        return out


def decode_attention_ref(q, k_cache, v_cache, pos, nq, nkv):
    """FP32 masked attention over the full capacity; graph safe, slower."""
    b, _, cap, d = k_cache.shape
    visible = torch.arange(cap, device=q.device) <= pos.to(torch.int64)
    qg = q.view(b, nkv, nq // nkv, d).to(torch.float32)
    scores = torch.matmul(qg, k_cache.to(torch.float32).transpose(-1, -2)) / math.sqrt(d)
    scores = scores.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_cache.to(torch.float32))
    return out.reshape(b, nq * d).to(q.dtype)
