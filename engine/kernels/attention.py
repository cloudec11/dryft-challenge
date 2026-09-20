"""Single-token GQA attention over a fixed-capacity KV cache.

One kernel, two specialisations:

``FUSE=True``   the program is handed the raw ``[B, (nq + 2*nkv) * D]`` QKV
                projection and rebuilds its own group's rotated Q plus this
                step's K and V from it, substitutes them for cache slot
                ``pos`` in registers, and -- in the one program whose split
                owns that slot -- stores them.  Head RMSNorm, RoPE, the cache
                write and attention are then a single launch.
``FUSE=False``  Q is already normalised, rotated and written, and the cache
                already holds this step's K/V.  Used by the reference decode
                path and while the fused one is being checked against it.

The sequence is split into ``splits`` chunks in the style of flash decoding.
Splitting is what fills the device when ``batch * kv_heads`` does not -- at
batch 1 that is 8 programs on 132 SMs -- and the cost has always been a second
launch to merge the partial softmaxes.  Here the merge happens in the same
launch: each split publishes its running ``(m, l, acc)``, bumps an atomic
counter for its ``(batch, kv head)``, and the split that reads the last ticket
merges them in split order and writes the output.  So the tile shape can be
chosen for the KV stream rather than to dodge a launch, at any batch.

At a large batch times context the KV read is the larger half of the step
(batch 32 over 2048 slots reads 9.8 GB against 8.05 GB of weights), so the
tile is still raced at warmup rather than guessed.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(
    q_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, ctr_ptr, out_ptr,
    stride_cb, stride_ch, stride_cs, chunk, num_splits, sm_scale_log2, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
    ONE_SPLIT: tl.constexpr, FIXUP: tl.constexpr, FUSE: tl.constexpr,
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
        # with FP32 accumulation, which is what SDPA does too.  ``dt`` is that
        # dtype, taken from the cache pointer rather than from the loaded
        # tile, so the cast is named the same way everywhere.
        acc = acc * alpha[:, None] + tl.dot(pr.to(dt), v)
        m_i = m_new

    if FUSE:
        owner = (start <= p) & (p < start + chunk)
        new_off = base + p.to(tl.int64) * stride_cs + offs_d
        # A vector mask, not a scalar one: the stored value is D wide.
        write = (offs_d < D) & owner
        tl.store(k_ptr + new_off, k_new, mask=write)
        tl.store(v_ptr + new_off, v_new, mask=write)

    out_off = b.to(tl.int64) * (NQ * D) + heads[:, None] * D + offs_d[None, :]
    if ONE_SPLIT:
        o = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + out_off, o, mask=h_mask[:, None])
    else:
        rows = (b * NQ + heads).to(tl.int64) * num_splits
        part = rows + split
        tl.store(po_ptr + part[:, None] * D + offs_d[None, :], acc, mask=h_mask[:, None],
                 cache_modifier=".cg")
        tl.store(pm_ptr + part, m_i, mask=h_mask, cache_modifier=".cg")
        tl.store(pl_ptr + part, l_i, mask=h_mask, cache_modifier=".cg")
        if FIXUP:
            # Same contract as the projection kernel's split-K fixup: publish,
            # sync the block, take a ticket.  The last ticket holder has
            # acquired every other split's stores and merges them in split
            # order, so the merge is deterministic.
            tl.debug_barrier()
            ticket = tl.atomic_add(ctr_ptr + b * NKV + kvh, 1, sem="acq_rel",
                                   scope="gpu")
            if ticket == num_splits - 1:
                tl.store(ctr_ptr + b * NKV + kvh, 0)
                offs_s = tl.arange(0, SPLITS)
                s_mask = (offs_s[None, :] < num_splits) & h_mask[:, None]
                idx = rows[:, None] + offs_s[None, :]
                m = tl.load(pm_ptr + idx, mask=s_mask, other=float("-inf"),
                            cache_modifier=".cg")
                l = tl.load(pl_ptr + idx, mask=s_mask, other=0.0,
                            cache_modifier=".cg")
                m_max = tl.max(m, axis=1)  # finite: split 0 always covers key 0
                # A split that saw nothing weighs exactly zero.
                w = tl.where(s_mask, tl.math.exp2(m - m_max[:, None]), 0.0)
                l_sum = tl.sum(l * w, axis=1)
                total = tl.zeros([BLOCK_H, D], dtype=tl.float32)
                for j in range(num_splits):
                    mj = tl.load(pm_ptr + rows + j, mask=h_mask,
                                 other=float("-inf"), cache_modifier=".cg")
                    wj = tl.where(h_mask, tl.math.exp2(mj - m_max), 0.0)
                    o = tl.load(po_ptr + (rows + j)[:, None] * D + offs_d[None, :],
                                mask=h_mask[:, None], other=0.0, cache_modifier=".cg")
                    total += o * wj[:, None]
                out = (total / l_sum[:, None]).to(out_ptr.dtype.element_ty)
                tl.store(out_ptr + out_off, out, mask=h_mask[:, None])


@triton.jit
def _attn_reduce_kernel(
    po_ptr, pm_ptr, pl_ptr, out_ptr, num_splits,
    NQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):
    """The fallback merge, in its own launch.  Same arithmetic as the fixup."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, D)
    s_mask = offs_s < num_splits
    part = (b * NQ + h).to(tl.int64) * num_splits + offs_s
    m = tl.load(pm_ptr + part, mask=s_mask, other=float("-inf"))
    l = tl.load(pl_ptr + part, mask=s_mask, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.math.exp2(m - m_max)
    l_sum = tl.sum(l * w, axis=0)
    o = tl.load(po_ptr + part[:, None] * D + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / l_sum
    tl.store(out_ptr + b.to(tl.int64) * (NQ * D) + h * D + offs_d,
             acc.to(out_ptr.dtype.element_ty))


def candidates(sm_count, batch, nkv, fixup=True):
    """Tile shapes worth racing, ordered by how well they fill the device.

    A program owns one ``(sequence, kv head)`` pair and one chunk of the
    sequence, so ``batch * nkv * splits`` programs have to cover the SMs: with
    one split that is 8 programs at batch 1 and 128 at batch 16.  The KV
    block width trades bytes in flight against the tail of a short chunk.
    """
    out = []
    groups = max(batch * nkv, 1)
    targets = [1]
    for mult in (1, 2, 4):
        want = sm_count * mult
        splits = max(1, -(-want // groups))
        if splits not in targets:
            targets.append(splits)
    for splits in targets:
        for block_n in (32, 64, 128, 256):
            if splits > 1 and not fixup:
                out.append((block_n, splits, False))
            out.append((block_n, splits, splits > 1 and fixup))
    # One split first: it needs neither partials nor a merge, so it is the
    # safest thing that can win.
    out.sort(key=lambda c: (c[1] > 1, not c[2], c[1], c[0]))
    seen, ordered = set(), []
    for cand in out:
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return ordered


class DecodeAttention:
    """Launcher owning the partial buffers for one shape.

    ``capacity`` is the KV cache's fixed slot count; ``splits`` divides it
    into equal chunks so the grid never changes shape, which is what lets the
    whole step be captured in a CUDA graph.
    """

    def __init__(self, batch, capacity, nq, nkv, head_dim, device,
                 block_n=64, splits=1, fixup=False, dtype=torch.bfloat16):
        self.batch, self.capacity = batch, capacity
        self.nq, self.nkv, self.head_dim = nq, nkv, head_dim
        self.group = nq // nkv
        self.block_h = max(16, triton.next_power_of_2(self.group))
        self.block_n = block_n
        self.splits = max(1, min(splits, capacity))
        self.chunk = -(-capacity // self.splits)
        # chunk is rounded up, so the last splits may cover nothing; drop them
        # rather than launch programs that only pay for a barrier.
        self.splits = -(-capacity // self.chunk)
        self.fixup = bool(fixup) and self.splits > 1
        self.splits_pow2 = triton.next_power_of_2(self.splits)
        self.scale_log2 = (head_dim ** -0.5) * math.log2(math.e)
        # One warp per 32 keys of the block: the K and V tiles are what the
        # program streams, and a wide block with four warps leaves each thread
        # holding more of the softmax than it has registers for.
        self.warps = 4 if block_n <= 64 else 8
        # Always real tensors, never None: an unused pointer argument still
        # has to have a type when the kernel is specialised.
        n = batch * nq * self.splits if self.splits > 1 else 1
        self.po = torch.zeros((n, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.zeros(n, dtype=torch.float32, device=device)
        self.pl = torch.zeros(n, dtype=torch.float32, device=device)
        self.ctr = torch.zeros(batch * nkv if self.fixup else 1,
                               dtype=torch.int32, device=device)

    def __repr__(self):
        tag = "" if self.splits == 1 else ("-fix" if self.fixup else "-red")
        return f"attn[{self.block_n}x{self.splits}{tag}]"

    def _run(self, q_or_qkv, k_cache, v_cache, pos, fuse, q_norm, k_norm, cos, sin,
             eps, out):
        one = self.splits == 1
        _attn_kernel[(self.batch, self.nkv, self.splits)](
            q_or_qkv, q_norm, k_norm, cos, sin, pos,
            k_cache, v_cache, self.po, self.pm, self.pl, self.ctr, out,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.chunk, self.splits, self.scale_log2, eps,
            NQ=self.nq, NKV=self.nkv, GROUP=self.group, D=self.head_dim,
            BLOCK_H=self.block_h, BLOCK_N=self.block_n, SPLITS=self.splits_pow2,
            ONE_SPLIT=one, FIXUP=self.fixup, FUSE=fuse,
            num_warps=self.warps, num_stages=2,
        )
        if not one and not self.fixup:
            _attn_reduce_kernel[(self.batch, self.nq)](
                self.po, self.pm, self.pl, out, self.splits,
                NQ=self.nq, D=self.head_dim, SPLITS=self.splits_pow2,
                num_warps=4, num_stages=1,
            )
        return out

    def plain(self, q, k_cache, v_cache, pos, out=None):
        """Q already normalised, rotated and appended to the cache."""
        if out is None:
            out = torch.empty((self.batch, self.nq * self.head_dim),
                              dtype=q.dtype, device=q.device)
        return self._run(q, k_cache, v_cache, pos, False, q, q, q, q, 0.0, out)

    def fused(self, qkv, q_norm, k_norm, cos, sin, pos, k_cache, v_cache, eps,
              out=None):
        """Raw packed projection in, attention out; the cache write included."""
        if out is None:
            out = torch.empty((self.batch, self.nq * self.head_dim),
                              dtype=qkv.dtype, device=qkv.device)
        return self._run(qkv, k_cache, v_cache, pos, True, q_norm, k_norm, cos, sin,
                         eps, out)
