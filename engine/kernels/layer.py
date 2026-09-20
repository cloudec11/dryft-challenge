"""One decode layer in one launch.

Why: v6 removed 36 launches per step (single-split attention skipping its
reduce) and batch-16 batch time fell 12 ms over 127 steps, so a launch here
costs ~2.8 us. The fused step issues five per layer, 185 per step, ~0.52 ms
of a step measured at 4.80 ms against a 2.81 ms byte floor -- and each launch
also refills the memory pipeline from empty. This kernel runs a whole layer
in one launch with grid-wide barriers between its stages, so a layer pays
that once instead of five times.

The arithmetic is lifted unchanged from ``fused.py`` and ``ops.py``: the same
rounding points, the same sum-of-squares hand-off between projections, the
same flash-decoding attention. Only the launch structure differs.

Barriers. Triton has no grid barrier, so each stage ends with an
arrive-and-wait over a per-(layer, stage) row of flags, zeroed once per step:

* arrive: store 1 into ``flags[slot, block]`` with release semantics;
* wait: spin until every one of the G entries is 1, with acquire semantics.

Two properties make this safe to try blind:

* **the spin is bounded.** A grid barrier deadlocks if the blocks are not
  co-resident, and a deadlock inside warmup would burn the 300 s budget and
  fail every workload. After ``SPIN_LIMIT`` attempts a block gives up and
  proceeds, which turns that failure into a wrong number instead of a hang --
  and a wrong number is exactly what the load-time check against the
  reference step catches, so the kernel is rejected and the five-launch step
  runs. The grid is one block per SM with modest shared memory, so all blocks
  are resident and the limit is never reached in the intended case.
* **the flag writes are idempotent.** Whether Triton emits the scalar store
  once per block or once per thread, every writer writes the same 1 to the
  same address, and the spin reads are atomic no-ops.
"""

import torch
import triton
import triton.language as tl

from kernels import ops

SPIN_LIMIT = 1 << 22  # ~4M polls: unreachable unless the barrier is broken
FLAG_BLOCK = 256  # >= grid, power of two for tl.arange


@triton.jit
def _arrive_and_wait(flag_ptr, slot, pid, G: tl.constexpr, FLAG_BLOCK: tl.constexpr):
    tl.debug_barrier()  # every warp in this block has finished the stage
    base = flag_ptr + slot * FLAG_BLOCK
    # Arrive with an atomic and no sem/scope keywords: those exist in newer
    # Triton than 3.1 is guaranteed to have, and an unsupported keyword is a
    # compile error that this engine would swallow as "unavailable". The
    # atomic still carries the ordering that makes this stage's stores
    # visible before the flag is.
    tl.atomic_xchg(base + pid, 1)
    offs = tl.arange(0, FLAG_BLOCK)
    mask = offs < G
    spins = 0
    done = 0
    while (done == 0) & (spins < SPIN_LIMIT):
        # volatile: the spin has to re-read memory every time, and a cached
        # load here is the classic way a hand-rolled barrier hangs.
        seen = tl.load(base + offs, mask=mask, other=1, volatile=True)
        done = tl.min(seen, axis=0)
        spins += 1
    tl.debug_barrier()


@triton.jit
def _layer_kernel(
    h_ptr, ssq_in_ptr, n_parts_in,
    in_w_ptr, post_w_ptr, qkv_w_ptr, qn_w_ptr, kn_w_ptr, o_w_ptr, gu_w_ptr, dn_w_ptr,
    qkv_ptr, attn_ptr, act_ptr,
    k_ptr, v_ptr, pos_ptr, cos_ptr, sin_ptr,
    po_ptr, pm_ptr, pl_ptr, ssq_a_ptr, ssq_b_ptr,
    flag_ptr, slot0,
    M, eps, sm_scale_log2,
    stride_cb, stride_ch, stride_cs, chunk, n_splits,
    HID: tl.constexpr, N_QKV: tl.constexpr, K_O: tl.constexpr, INTER: tl.constexpr,
    NQ: tl.constexpr, NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BATCH: tl.constexpr, SPLITS: tl.constexpr, SPLITS_P: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    BN_QKV: tl.constexpr, BN_O: tl.constexpr, BN_GU: tl.constexpr, BN_DN: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, PARTS: tl.constexpr,
    G: tl.constexpr, SS_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BK)
    dt = h_ptr.dtype.element_ty

    # ---- stage 0: input RMSNorm (from the previous layer's partials) + QKV
    offs_p = tl.arange(0, PARTS)
    ss = tl.load(ssq_in_ptr + offs_p[:, None] * SS_STRIDE + offs_m[None, :],
                 mask=(offs_p[:, None] < n_parts_in) & m_mask[None, :], other=0.0)
    rstd = tl.math.rsqrt(tl.sum(ss, axis=0) / HID + eps)
    for tile in range(pid, N_QKV // BN_QKV, G):
        offs_n = tile * BN_QKV + tl.arange(0, BN_QKV)
        x_ptrs = h_ptr + offs_m[:, None] * HID + offs_k[None, :]
        w_ptrs = qkv_w_ptr + offs_n[None, :].to(tl.int64) * HID + offs_k[:, None]
        acc = tl.zeros((BM, BN_QKV), dtype=tl.float32)
        for k0 in range(0, HID, BK):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            nw = tl.load(in_w_ptr + k0 + offs_k)
            w = tl.load(w_ptrs)
            # bf16(w_norm * bf16(h * rstd)): the reference's rounding points.
            xn = (x.to(tl.float32) * rstd[:, None]).to(dt)
            xn = (nw[None, :].to(tl.float32) * xn.to(tl.float32)).to(dt)
            acc += tl.dot(xn, w)
            x_ptrs += BK
            w_ptrs += BK
        tl.store(qkv_ptr + offs_m[:, None] * N_QKV + offs_n[None, :], acc.to(dt),
                 mask=m_mask[:, None])
    _arrive_and_wait(flag_ptr, slot0 + 0, pid, G, FLAG_BLOCK)

    # ---- stage 1: Q/K head norm + RoPE + KV write + flash decoding
    p = tl.load(pos_ptr)
    seq_len = p + 1
    offs_d = tl.arange(0, D)
    partner = (offs_d + D // 2) % D
    first_half = offs_d < D // 2
    offs_h = tl.arange(0, BLOCK_H)
    h_ok = offs_h < GROUP
    c = tl.load(cos_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)
    s = tl.load(sin_ptr + p.to(tl.int64) * D + offs_d).to(tl.float32)
    for unit in range(pid, BATCH * NKV * SPLITS, G):
        b = unit // (NKV * SPLITS)
        kvh = (unit // SPLITS) % NKV
        split = unit % SPLITS
        start = split * chunk
        end = tl.minimum(start + chunk, seq_len)
        row = qkv_ptr + b * N_QKV
        heads = kvh * GROUP + offs_h

        xq = tl.load(row + heads[:, None] * D + offs_d[None, :], mask=h_ok[:, None],
                     other=0.0).to(tl.float32)
        xqp = tl.load(row + heads[:, None] * D + partner[None, :], mask=h_ok[:, None],
                      other=0.0).to(tl.float32)
        rq = tl.math.rsqrt(tl.sum(xq * xq, axis=1) / D + eps)
        wq = tl.load(qn_w_ptr + offs_d).to(tl.float32)
        wqp = tl.load(qn_w_ptr + partner).to(tl.float32)
        yq = (wq[None, :] * (xq * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        yqp = (wqp[None, :] * (xqp * rq[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        rotq = tl.where(first_half[None, :], -yqp, yqp)
        q = ((yq * c[None, :]).to(dt).to(tl.float32)
             + (rotq * s[None, :]).to(dt).to(tl.float32)).to(dt)

        kcol = row + (NQ + kvh) * D
        xk = tl.load(kcol + offs_d).to(tl.float32)
        xkp = tl.load(kcol + partner).to(tl.float32)
        rk = tl.math.rsqrt(tl.sum(xk * xk, axis=0) / D + eps)
        wk = tl.load(kn_w_ptr + offs_d).to(tl.float32)
        wkp = tl.load(kn_w_ptr + partner).to(tl.float32)
        yk = (wk * (xk * rk).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        ykp = (wkp * (xkp * rk).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        rotk = tl.where(first_half, -ykp, ykp)
        k_new = ((yk * c).to(dt).to(tl.float32) + (rotk * s).to(dt).to(tl.float32)).to(dt)
        v_new = tl.load(row + (NQ + NKV + kvh) * D + offs_d)

        base = b * stride_cb + kvh * stride_ch
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_a = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_ok = offs_n < end
            is_new = (offs_n == p)[:, None]
            kv_off = base + offs_n[:, None].to(tl.int64) * stride_cs + offs_d[None, :]
            k = tl.load(k_ptr + kv_off, mask=n_ok[:, None], other=0.0)
            k = tl.where(is_new, k_new[None, :], k)
            sc = tl.dot(q, tl.trans(k)) * sm_scale_log2
            sc = tl.where(n_ok[None, :], sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            pr = tl.math.exp2(sc - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pr, axis=1)
            vv = tl.load(v_ptr + kv_off, mask=n_ok[:, None], other=0.0)
            vv = tl.where(is_new, v_new[None, :], vv)
            acc_a = acc_a * alpha[:, None] + tl.dot(pr.to(vv.dtype), vv)
            m_i = m_new
        owner = (start <= p) & (p < start + chunk)
        new_off = base + p.to(tl.int64) * stride_cs + offs_d
        tl.store(k_ptr + new_off, k_new, mask=(offs_d < D) & owner)
        tl.store(v_ptr + new_off, v_new, mask=(offs_d < D) & owner)
        if SPLITS == 1:
            o = (acc_a / l_i[:, None]).to(dt)
            tl.store(attn_ptr + (b * NQ + heads)[:, None] * D + offs_d[None, :],
                     o, mask=h_ok[:, None])
        else:
            part = (b * NQ + heads).to(tl.int64) * n_splits + split
            tl.store(po_ptr + part[:, None] * D + offs_d[None, :], acc_a, mask=h_ok[:, None])
            tl.store(pm_ptr + part, m_i, mask=h_ok)
            tl.store(pl_ptr + part, l_i, mask=h_ok)
    _arrive_and_wait(flag_ptr, slot0 + 1, pid, G, FLAG_BLOCK)

    # ---- stage 2: attention reduce (only when the sequence was split)
    if SPLITS > 1:
        offs_s = tl.arange(0, SPLITS_P)
        for unit in range(pid, BATCH * NQ, G):
            b = unit // NQ
            head = unit % NQ
            part = offs_s + (b * NQ + head) * n_splits
            s_ok = offs_s < n_splits
            mm = tl.load(pm_ptr + part, mask=s_ok, other=float("-inf"))
            ll = tl.load(pl_ptr + part, mask=s_ok, other=0.0)
            m_max = tl.max(mm, axis=0)
            wgt = tl.math.exp2(mm - m_max)
            l_sum = tl.sum(ll * wgt, axis=0)
            oo = tl.load(po_ptr + part[:, None] * D + offs_d[None, :], mask=s_ok[:, None],
                         other=0.0)
            red = tl.sum(oo * wgt[:, None], axis=0) / l_sum
            tl.store(attn_ptr + (b * NQ + head) * D + offs_d, red.to(dt))
        _arrive_and_wait(flag_ptr, slot0 + 2, pid, G, FLAG_BLOCK)

    # ---- stage 3: O projection, residual add in place, sums of squares
    for tile in range(pid, HID // BN_O, G):
        offs_n = tile * BN_O + tl.arange(0, BN_O)
        x_ptrs = attn_ptr + offs_m[:, None] * K_O + offs_k[None, :]
        w_ptrs = o_w_ptr + offs_n[None, :].to(tl.int64) * K_O + offs_k[:, None]
        acc = tl.zeros((BM, BN_O), dtype=tl.float32)
        for _ in range(0, K_O, BK):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BK
            w_ptrs += BK
        off = offs_m[:, None] * HID + offs_n[None, :]
        d = acc.to(dt).to(tl.float32)
        r = tl.load(h_ptr + off, mask=m_mask[:, None], other=0.0).to(tl.float32)
        hh = (r + d).to(dt)
        tl.store(h_ptr + off, hh, mask=m_mask[:, None])
        hf = tl.where(m_mask[:, None], hh.to(tl.float32), 0.0)
        tl.store(ssq_a_ptr + tile * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1), mask=m_mask)
    _arrive_and_wait(flag_ptr, slot0 + 3, pid, G, FLAG_BLOCK)

    # ---- stage 4: post-attention RMSNorm + gate/up + SwiGLU
    parts_o = HID // BN_O
    ss = tl.load(ssq_a_ptr + offs_p[:, None] * SS_STRIDE + offs_m[None, :],
                 mask=(offs_p[:, None] < parts_o) & m_mask[None, :], other=0.0)
    rstd2 = tl.math.rsqrt(tl.sum(ss, axis=0) / HID + eps)
    for tile in range(pid, INTER // BN_GU, G):
        offs_n = tile * BN_GU + tl.arange(0, BN_GU)
        x_ptrs = h_ptr + offs_m[:, None] * HID + offs_k[None, :]
        w_ptrs = gu_w_ptr + offs_n[None, :].to(tl.int64) * HID + offs_k[:, None]
        wu_ptrs = w_ptrs + INTER * HID
        acc = tl.zeros((BM, BN_GU), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN_GU), dtype=tl.float32)
        for k0 in range(0, HID, BK):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            nw = tl.load(post_w_ptr + k0 + offs_k)
            xn = (x.to(tl.float32) * rstd2[:, None]).to(dt)
            xn = (nw[None, :].to(tl.float32) * xn.to(tl.float32)).to(dt)
            acc += tl.dot(xn, tl.load(w_ptrs))
            acc_u += tl.dot(xn, tl.load(wu_ptrs))
            x_ptrs += BK
            w_ptrs += BK
            wu_ptrs += BK
        g = acc.to(dt).to(tl.float32)
        u = acc_u.to(dt).to(tl.float32)
        a = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
        tl.store(act_ptr + offs_m[:, None] * INTER + offs_n[None, :], (a * u).to(dt),
                 mask=m_mask[:, None])
    _arrive_and_wait(flag_ptr, slot0 + 4, pid, G, FLAG_BLOCK)

    # ---- stage 5: down projection, residual add in place, sums of squares
    for tile in range(pid, HID // BN_DN, G):
        offs_n = tile * BN_DN + tl.arange(0, BN_DN)
        x_ptrs = act_ptr + offs_m[:, None] * INTER + offs_k[None, :]
        w_ptrs = dn_w_ptr + offs_n[None, :].to(tl.int64) * INTER + offs_k[:, None]
        acc = tl.zeros((BM, BN_DN), dtype=tl.float32)
        for _ in range(0, INTER, BK):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BK
            w_ptrs += BK
        off = offs_m[:, None] * HID + offs_n[None, :]
        d = acc.to(dt).to(tl.float32)
        r = tl.load(h_ptr + off, mask=m_mask[:, None], other=0.0).to(tl.float32)
        hh = (r + d).to(dt)
        tl.store(h_ptr + off, hh, mask=m_mask[:, None])
        hf = tl.where(m_mask[:, None], hh.to(tl.float32), 0.0)
        tl.store(ssq_b_ptr + tile * SS_STRIDE + offs_m, tl.sum(hf * hf, axis=1), mask=m_mask)
    # No trailing barrier: the next layer is a new launch, which is one.


class LayerPlan:
    """Tile widths, grid and barrier flags for the one-launch layer."""

    def __init__(self, engine, batch, capacity, block_m, grid=None):
        dev = engine.device
        props = torch.cuda.get_device_properties(dev)
        self.hidden = engine.norm_w.shape[0]
        self.inter = engine.layers[0].down.shape[1]
        self.n_qkv = engine.layers[0].qkv.shape[0]
        self.k_o = engine.layers[0].o.shape[1]
        self.batch = batch
        self.bm = block_m
        self.bk = 128
        # One block per SM: the barrier needs every block resident, and the
        # stages are grid-strided so any grid is correct, only faster or
        # slower. Never more blocks than the widest stage has tiles.
        self.grid = min(grid or props.multi_processor_count, FLAG_BLOCK)
        self.bn_qkv, self.bn_o, self.bn_gu, self.bn_dn = 32, 16, 64, 16
        self.block_h = max(16, triton.next_power_of_2(engine.nq // engine.nkv))
        # Its own attention instance: the partial buffers have to be sized by
        # the same split count the kernel loops over, and targeting the grid
        # keeps one attention unit per block.
        self.attn = ops.DecodeAttention(batch, capacity, engine.nq, engine.nkv,
                                        engine.head_dim, dev, target_programs=self.grid)
        self.block_n = self.attn.block_n
        self.splits = self.attn.splits
        self.chunk = self.attn.chunk
        self.stages = 6
        self.flags = torch.zeros((engine.n_layers * self.stages, FLAG_BLOCK),
                                 dtype=torch.int32, device=dev)
        self.attn_buf = torch.empty((batch, engine.nq * engine.head_dim),
                                    dtype=engine.dtype, device=dev)
        # Partials the down projection leaves for the next layer's RMSNorm,
        # and for the LM head's after the last layer.
        self.parts_dn = self.hidden // self.bn_dn
        self.parts_block = max(2, 1 << (max(self.parts_dn, self.hidden // self.bn_o) - 1).bit_length())

    def smem_bytes(self):
        widest = max(self.bn_qkv, self.bn_o, 2 * self.bn_gu, self.bn_dn)
        return 3 * (self.bm * self.bk + self.bk * widest) * 2


def layer_forward(engine, st, plan, layer_index, layer, h, ssq_in, n_parts_in,
                  qkv_buf, attn_buf, act_buf, ssq_a, ssq_b):
    """Run one decoder layer as a single launch."""
    attn = plan.attn
    _layer_kernel[(plan.grid,)](
        h, ssq_in, n_parts_in,
        layer.in_w, layer.post_w, layer.qkv, layer.q_norm, layer.k_norm,
        layer.o, layer.gate_up, layer.down,
        qkv_buf, attn_buf, act_buf,
        st.k_cache[layer_index], st.v_cache[layer_index], st.pos, engine.cos, engine.sin,
        attn.po, attn.pm, attn.pl, ssq_a, ssq_b,
        plan.flags, layer_index * plan.stages,
        plan.batch, engine.eps, attn.sm_scale_log2,
        st.k_cache.stride(1), st.k_cache.stride(2), st.k_cache.stride(3),
        plan.chunk, plan.splits,
        HID=plan.hidden, N_QKV=plan.n_qkv, K_O=plan.k_o, INTER=plan.inter,
        NQ=engine.nq, NKV=engine.nkv, GROUP=engine.nq // engine.nkv, D=engine.head_dim,
        BATCH=plan.batch, SPLITS=plan.splits,
        SPLITS_P=max(2, triton.next_power_of_2(plan.splits)),
        BM=plan.bm, BK=plan.bk,
        BN_QKV=plan.bn_qkv, BN_O=plan.bn_o, BN_GU=plan.bn_gu, BN_DN=plan.bn_dn,
        BLOCK_H=plan.block_h, BLOCK_N=plan.block_n, PARTS=plan.parts_block,
        G=plan.grid, SS_STRIDE=plan.bm,
        num_warps=4, num_stages=3,
    )
