"""Speculative decoding: n-gram drafts from the sequence's own history,
verified exactly.

Why it can win: a decode step is weight-bound. Streaming the 8.05 GB of
weights costs the same whether one token or five ride along, so verifying
K+1 tokens per sequence costs ~1.1x a single-token step and every accepted
draft is nearly free. Correctness is by construction -- a token is emitted
only when it equals the argmax the model itself produces at that position,
teacher-forced on the tokens we already emitted -- so this is the one
technique here that changes the token count per step without touching the
2.0-logit budget at all.

Per iteration, for every sequence b with cache length L[b]:

* draft: take the last NGRAM tokens of the sequence's own history, find the
  most recent earlier occurrence, and copy the K tokens that followed it.
* verify: run the fused decode step over T = K+1 rows per sequence, feeding
  [current token, draft...] at positions L[b] .. L[b]+K, and take the argmax
  at each row. Row t's argmax is the true greedy token at position L[b]+t
  *provided* every draft before it was right, which is exactly the accept
  rule below.
* accept: a = the number of leading drafts that match their argmax; emit
  a+1 tokens (rows 0..a); advance L[b] by a+1, since the K/V written for
  rows 0..a are the K/V of tokens that really are in the sequence.

Positions are per sequence (a device vector, not a scalar), because
acceptance differs per sequence. Slots past L[b] hold K/V of rejected
drafts; nothing reads them, and the next iteration overwrites them.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ngram_draft_kernel(
    hist_ptr, lens_ptr, ids_ptr, draft_ptr, cap,
    NGRAM: tl.constexpr, K: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr,
    CAP: tl.constexpr,
):
    # History of sequence b is hist[b, 0 .. L] inclusive: L = lens[b] slots
    # are in the KV cache and hist[b, L] is the token not yet fed (the one
    # this iteration feeds as row 0).
    b = tl.program_id(0)
    L = tl.load(lens_ptr + b)
    base = b.to(tl.int64) * cap
    tl.store(ids_ptr + b * T, tl.load(hist_ptr + base + L))  # row 0: current token

    # Most recent j < L - NGRAM + 1 with hist[j .. j+NGRAM-1] == the last
    # NGRAM tokens. Scanning forward and keeping the max index gives the most
    # recent match, which is the better predictor in practice.
    best = -1
    limit = L - NGRAM + 1  # j must leave a following token at j+NGRAM <= L
    # ``L`` varies between sequences after the first speculative iteration.
    # Keep the loop bound a compile-time capacity instead of using a runtime
    # Python-range endpoint: Triton 3.1 can then compile one stable graph
    # specialization for the shape, and the mask excludes both unwritten
    # cache slots and positions beyond this sequence's valid history.
    for j0 in range(0, CAP, BLOCK):
        offs = j0 + tl.arange(0, BLOCK)
        ok = offs < limit
        for i in tl.static_range(NGRAM):
            # Clamped: a sequence shorter than the n-gram would index before
            # the row, and an unmasked out-of-bounds read takes the context
            # down rather than failing softly.
            pat_at = tl.maximum(L - NGRAM + 1 + i, 0)
            pat = tl.load(hist_ptr + base + pat_at)
            tok = tl.load(hist_ptr + base + offs + i, mask=ok, other=-1)
            ok = ok & (tok == pat)
        best = tl.maximum(best, tl.max(tl.where(ok, offs, -1), axis=0))

    for t in tl.static_range(K):
        idx = best + NGRAM + t
        ok = (best >= 0) & (idx <= L)
        tok = tl.load(hist_ptr + base + idx, mask=ok, other=0)
        tl.store(draft_ptr + b * K + t, tok)
        tl.store(ids_ptr + b * T + 1 + t, tok)


def ngram_draft(hist, lens, ids, draft, ngram, k, t):
    """Fill ``ids`` ``[B*T]`` with [current, drafts...] per sequence and
    ``draft`` ``[B, K]`` with the drafts alone."""
    batch, cap = hist.shape
    _ngram_draft_kernel[(batch,)](
        hist, lens, ids, draft, cap,
        NGRAM=ngram, K=k, T=t, BLOCK=1024, CAP=cap, num_warps=4,
    )


@triton.jit
def _spec_attn_kernel(
    q_ptr, k_ptr, v_ptr, lens_ptr, out_ptr,
    stride_cb, stride_ch, stride_cs, sm_scale_log2,
    T: tl.constexpr, NQ: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, CAP: tl.constexpr,
):
    # One program per (sequence, kv head), BLOCK_R rows = T queries x GROUP
    # q heads. Query row t sees cache slots 0 .. L+t: the prefix plus the
    # tokens this iteration fed before it. K/V for all T rows are already in
    # the cache (qkv_post wrote them), so this kernel only reads.
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    L = tl.load(lens_ptr + b)
    r = tl.arange(0, BLOCK_R)
    t_of = r // GROUP
    g_of = r % GROUP
    r_mask = r < T * GROUP
    offs_d = tl.arange(0, D)
    rows = b * T + t_of
    q = tl.load(
        q_ptr + rows[:, None].to(tl.int64) * (NQ * D) + (kvh * GROUP + g_of)[:, None] * D
        + offs_d[None, :],
        mask=r_mask[:, None], other=0.0,
    )
    last = L + t_of  # highest visible slot for each row

    base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    m_i = tl.full([BLOCK_R], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc = tl.zeros([BLOCK_R, D], dtype=tl.float32)
    end = L + T
    # As with the draft lookup, lengths vary across sequences.  A
    # capacity-specialized loop is graph-safe on Triton 3.1; masked tail
    # blocks are mathematically empty and do not expose unwritten cache
    # entries to the softmax.
    for n0 in range(0, CAP, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv_off = base + offs_n[:, None].to(tl.int64) * stride_cs + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=(offs_n < end)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale_log2
        s = tl.where(offs_n[None, :] <= last[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=(offs_n < end)[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * (NQ * D) + (kvh * GROUP + g_of)[:, None] * D
        + offs_d[None, :], o, mask=r_mask[:, None],
    )


def spec_attention(q, k_cache, v_cache, lens, batch, t, nq, nkv, head_dim, sm_scale_log2,
                   block_n=64, num_warps=4, num_stages=3):
    """Causal attention for T new rows per sequence over per-sequence cache
    lengths. ``q`` is ``[B*T, nq*D]``; returns the same shape."""
    group = nq // nkv
    out = torch.empty_like(q)
    _spec_attn_kernel[(batch, nkv)](
        q, k_cache, v_cache, lens, out,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), sm_scale_log2,
        T=t, NQ=nq, GROUP=group, D=head_dim,
        BLOCK_R=max(16, triton.next_power_of_2(t * group)), BLOCK_N=block_n,
        CAP=k_cache.shape[2],
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _accept_kernel(
    out_tok_ptr, draft_ptr, hist_ptr, lens_ptr, naccept_ptr, cap, stop_len,
    K: tl.constexpr, T: tl.constexpr,
):
    # a = number of leading drafts that match the model's own argmax at the
    # same position. Rows 0..a are then all genuine greedy tokens, so they
    # are emitted, appended to the history and counted into the cache length.
    b = tl.program_id(0)
    L = tl.load(lens_ptr + b)
    base = b.to(tl.int64) * cap
    a = 0
    run = 1  # still matching so far
    for t in tl.static_range(K):
        got = tl.load(out_tok_ptr + b * T + t)
        want = tl.load(draft_ptr + b * K + t)
        run = run * tl.where(got == want, 1, 0)
        a += run
    # A sequence that already has all its tokens stops advancing, so its
    # writes stay inside the cache; the loop runs until the slowest is done.
    live = tl.where(L < stop_len, 1, 0)
    a = tl.where(live == 1, a, 0)
    # hist[L] already holds row 0's input token; rows 0..a are new tokens, so
    # they land at L+1 .. L+1+a.
    for t in tl.static_range(T):
        tok = tl.load(out_tok_ptr + b * T + t)
        idx = L + 1 + t
        tl.store(hist_ptr + base + idx, tok, mask=(t <= a) & (idx < cap) & (live == 1))
    tl.store(naccept_ptr + b, a)
    tl.store(lens_ptr + b, L + tl.where(live == 1, a + 1, 0))


def accept(out_tok, draft, hist, lens, naccept, k, t, stop_len):
    """Per sequence: count accepted drafts, append the emitted tokens to the
    history and advance the cache length. ``stop_len`` freezes a sequence
    that already has every token it needs."""
    batch, cap = hist.shape
    _accept_kernel[(batch,)](
        out_tok, draft, hist, lens, naccept, cap, stop_len, K=k, T=t, num_warps=1,
    )
