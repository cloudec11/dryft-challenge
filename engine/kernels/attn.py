"""Single-token GQA attention over a fixed-capacity KV cache.

One kernel, two specialisations:

``FUSE=True``   the program is handed the raw ``[B, (nq + 2*nkv) * D]`` QKV
                projection and rebuilds its own group's rotated Q plus this
                step's K and V from it, substitutes them for cache slot
                ``pos`` in registers, and -- in the one program whose split
                owns that slot -- stores them. Head RMSNorm, RoPE, the cache
                write and attention are then a single launch.
``FUSE=False``  Q is already normalised, rotated and written, and the cache
                already holds this step's K/V. Used by the reference decode
                path and while the fused one is being checked against it.

The sequence is split into ``splits`` chunks in the style of flash decoding.
With one split the program already holds the whole running softmax, so it
divides by ``l_i`` itself and the reduce kernel is not launched at all --
36 fewer launches per step, which at batch 16 was worth ~2% of the step.

At a large batch times context the KV read is the larger half of the step
(batch 32 over 2048 slots reads 9.8 GB against 8.05 GB of weights), so the
tile shape is raced at warmup rather than guessed.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(
    q_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, out_ptr,
    stride_cb, stride_ch, stride_cs, chunk, num_splits, sm_scale_log2, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    ONE_SPLIT: tl.constexpr, FUSE: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    p = tl.load(pos_ptr)
    seq_len = p + 1  # this step's K/V is visible to this step
    start = split * chunk
    end = tl.minimum(start + chunk, seq_len)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    dt = k_ptr.dtype.element_ty

    if FUSE:
        # Rebuild q, k and v for this program's kv head from the raw
        # projection: a few hundred bytes, and it removes a whole launch.
        partner = (offs_d + D // 2) % D
        first_half = offs_d < D // 2
        row = q_ptr + b.to(tl.int64) * ((NQ + 2 * NKV) * D)
        c = tl.load(cos_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)
        s = tl.load(sin_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)

        xq = tl.load(row + heads[:, None] * D + offs_d[None, :],
                     mask=h_mask[:, None], other=0.0).to(tl.float32)
        xqp = tl.load(row + heads[:, None] * D + partner[None, :],
                      mask=h_mask[:, None], other=0.0).to(tl.float32)
        rq = tl.math.rsqrt(tl.sum(xq * xq, axis=1) / D + eps)
        wq = tl.load(qw_ptr + offs_d).to(tl.float32)
        wqp = tl.load(qw_ptr + partner).to(tl.float32)
        # Qwen3RMSNorm rounds after the normalise and again after the weight.
        yq = (wq[None, :] * (xq * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        yqp = (wqp[None, :] * (xqp * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        # rotate_half: the first half takes -x2, the second half takes x1.
        rotq = tl.where(first_half[None, :], -yqp, yqp)
        q = ((yq * c[None, :]).to(dt).to(tl.float32)
             + (rotq * s[None, :]).to(dt).to(tl.float32)).to(dt)

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
    else:
        q = tl.load(q_ptr + b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :],
                    mask=h_mask[:, None], other=0.0)

    base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        kv_off = base + offs_n[:, None].to(tl.int64) * stride_cs + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        if FUSE:
            # Splits cover disjoint ranges, so no program reads a slot that
            # another program of this launch is writing.
            is_new = (offs_n == p)[:, None]
            k = tl.where(is_new, k_new[None, :], k)
        sc = tl.dot(q, tl.trans(k)) * sm_scale_log2
        sc = tl.where(n_mask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        pr = tl.math.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, axis=1)
        v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        if FUSE:
            v = tl.where(is_new, v_new[None, :], v)
        # As in FlashAttention, P enters the PV product in the value dtype
        # with FP32 accumulation, which is what SDPA does too.
        acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
        m_i = m_new

    if FUSE:
        owner = (start <= p) & (p < start + chunk)
        new_off = base + p.to(tl.int64) * stride_cs + offs_d
        # A vector mask, not a scalar one: the stored value is D wide.
        write = (offs_d < D) & owner
        tl.store(k_ptr + new_off, k_new, mask=write)
        tl.store(v_ptr + new_off, v_new, mask=write)

    if ONE_SPLIT:
        o = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :],
                 o, mask=h_mask[:, None])
    else:
        part = (b * NQ + heads).to(tl.int64) * num_splits + split
        tl.store(po_ptr + part[:, None] * D + offs_d[None, :], acc, mask=h_mask[:, None])
        tl.store(pm_ptr + part, m_i, mask=h_mask)
        tl.store(pl_ptr + part, l_i, mask=h_mask)


@triton.jit
def _attn_reduce_kernel(
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
    m_max = tl.max(m, axis=0)  # finite: split 0 always covers key 0
    w = tl.math.exp2(m - m_max)  # a split that saw nothing weighs exactly zero
    l_sum = tl.sum(l * w, axis=0)
    o = tl.load(po_ptr + part[:, None] * D + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / l_sum
    tl.store(out_ptr + b.to(tl.int64) * (NQ * D) + h * D + offs_d,
             acc.to(out_ptr.dtype.element_ty))


def candidates(sm_count):
    """``(BLOCK_N, target programs, num_warps, num_stages)`` to race.

    ``target programs`` is turned into a split count by
    :class:`DecodeAttention`. One split is offered explicitly because it
    skips the reduce launch, and is worth taking as soon as
    ``batch * kv_heads`` fills the device on its own.
    """
    s = sm_count
    return (
        (64, 2 * s, 4, 3),      # what v1-v10 ran; the incumbent
        (128, 2 * s, 4, 3),
        (64, 4 * s, 4, 3),
        (128, s, 8, 3),
        (64, s, 4, 4),
        (256, 2 * s, 8, 3),
        (128, 2 * s, 8, 4),
        (64, 1, 4, 3),          # one split: no reduce kernel
        (128, 1, 8, 3),
        (256, 1, 8, 3),
    )


class DecodeAttention:
    """Attention of one new token per sequence over ``[0, pos]``.

    The split layout depends only on batch and capacity, never on the current
    position, so one instance serves every step of a captured graph.
    """

    def __init__(self, batch, capacity, nq, nkv, head_dim, device,
                 target_programs=264, block_n=64, num_warps=4, num_stages=3):
        self.batch, self.nq, self.nkv, self.d = batch, nq, nkv, head_dim
        self.group = nq // nkv
        self.block_n = block_n
        self.num_warps, self.num_stages = num_warps, num_stages
        max_splits = triton.cdiv(capacity, block_n)
        want = max(1, triton.cdiv(target_programs, batch * nkv))
        splits = min(want, max_splits)
        self.chunk = triton.cdiv(triton.cdiv(capacity, splits), block_n) * block_n
        self.splits = triton.cdiv(capacity, self.chunk)
        self.sm_scale_log2 = (1.0 / math.sqrt(head_dim)) * 1.4426950408889634
        n_part = batch * nq * self.splits
        self.po = torch.empty((n_part, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.empty((n_part,), dtype=torch.float32, device=device)
        self.pl = torch.empty((n_part,), dtype=torch.float32, device=device)

    def __repr__(self):
        return f"bn{self.block_n}x{self.splits}w{self.num_warps}s{self.num_stages}"

    def _run(self, q_or_qkv, k_cache, v_cache, pos, fuse, q_norm, k_norm, cos, sin, eps, out):
        one = self.splits == 1
        _attn_kernel[(self.batch, self.nkv, self.splits)](
            q_or_qkv, q_norm, k_norm, cos, sin, pos,
            k_cache, v_cache, self.po, self.pm, self.pl, out,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.chunk, self.splits, self.sm_scale_log2, eps,
            NQ=self.nq, NKV=self.nkv, GROUP=self.group, D=self.d,
            BLOCK_H=max(16, triton.next_power_of_2(self.group)), BLOCK_N=self.block_n,
            ONE_SPLIT=one, FUSE=fuse,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if not one:
            _attn_reduce_kernel[(self.batch, self.nq)](
                self.po, self.pm, self.pl, out, self.splits,
                NQ=self.nq, D=self.d,
                SPLITS=max(2, triton.next_power_of_2(self.splits)), num_warps=4,
            )
        return out

    def plain(self, q, k_cache, v_cache, pos, out=None):
        """Q already normalised, rotated and written into the cache. The
        weight and table pointers are unused in this specialisation, so q
        stands in for them and the launch signature stays the same."""
        if out is None:
            out = torch.empty((self.batch, self.nq * self.d), dtype=q.dtype, device=q.device)
        return self._run(q, k_cache, v_cache, pos, False, q, q, q, q, 0.0, out)

    def fused(self, qkv, q_norm, k_norm, cos, sin, pos, k_cache, v_cache, eps, out=None):
        """Raw QKV in, attention out; the cache write happens on the way."""
        if out is None:
            out = torch.empty((self.batch, self.nq * self.d), dtype=qkv.dtype,
                              device=qkv.device)
        return self._run(qkv, k_cache, v_cache, pos, True, q_norm, k_norm, cos, sin, eps, out)
