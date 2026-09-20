"""PyTorch twins of every Triton kernel: the load-time judge, and the net.

Each one is the same arithmetic written the obvious way, with the casts in
the same places Transformers 4.51.3 puts them.  The engine runs both on
random inputs of the real shape before it trusts a kernel, and if a kernel
disagrees or fails to compile, the twin takes its place for the whole run.
That is why the engine can ship a new kernel at all: the failure mode is
"slower than it could have been", never "wrong answer".

The twins are also the readable specification.  When a comment in a Triton
kernel says "the reference rounds here", this is the reference.
"""

import torch


def rms_norm(x, weight, eps):
    """Qwen3RMSNorm: reduce in FP32, round to the input dtype, then weight."""
    dt = x.dtype
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * (xf * rstd).to(dt).float()).to(dt)


def embed(ids, emb_w, h, ssq):
    rows = emb_w[ids.long()]
    h.copy_(rows)
    ssq[0, : ids.shape[0]] = rows.float().pow(2).sum(-1)


def add_rms_norm(x, residual, weight, eps, out=None):
    if residual is not None:
        h = (residual.float() + x.float()).to(x.dtype)
        residual.copy_(h)
    else:
        h = x
    y = rms_norm(h, weight, eps)
    if out is not None:
        out.copy_(y)
        return out
    return y


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x, cos, sin):
    dt = x.dtype
    return (((x.float() * cos.float()).to(dt).float()
             + (_rotate_half(x).float() * sin.float()).to(dt).float())).to(dt)


def qkv_post(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv,
             eps, out=None, row0=0):
    m, d = qkv.shape[0], q_weight.shape[0]
    assert row0 % T == 0 and m % T == 0, "the twin only covers whole sequences"
    batch = m // T
    b0 = row0 // T
    start = int(pos.item())
    parts = qkv.view(m, -1, d)
    q = rms_norm(parts[:, :nq], q_weight, eps).view(batch, T, nq, d)
    k = rms_norm(parts[:, nq:nq + nkv], k_weight, eps).view(batch, T, nkv, d)
    v = parts[:, nq + nkv:].reshape(batch, T, nkv, d)
    c = cos[start:start + T].view(1, T, 1, d)
    s = sin[start:start + T].view(1, T, 1, d)
    q = _rope(q, c, s)
    k = _rope(k, c, s)
    k_cache[b0:b0 + batch, :, start:start + T] = k.transpose(1, 2)
    v_cache[b0:b0 + batch, :, start:start + T] = v.transpose(1, 2)
    q = q.reshape(m, nq * d)
    if out is not None:
        out.copy_(q)
        return out
    return q


def silu_mul(gate_up, out=None):
    dt = gate_up.dtype
    inter = gate_up.shape[1] // 2
    g = gate_up[:, :inter].float()
    u = gate_up[:, inter:].float()
    act = (g * torch.sigmoid(g)).to(dt).float()
    y = (act * u).to(dt)
    if out is not None:
        out.copy_(y)
        return out
    return y


def decode_attention(q, k_cache, v_cache, pos, nq, nkv, out=None):
    """``q`` is ``[B, nq * D]``; the cache already holds this step's K and V."""
    batch = q.shape[0]
    d = k_cache.shape[-1]
    length = int(pos.item()) + 1
    group = nq // nkv
    qh = q.view(batch, nq, 1, d).float()
    k = k_cache[:, :, :length].repeat_interleave(group, dim=1).float()
    v = v_cache[:, :, :length].repeat_interleave(group, dim=1)
    scores = torch.matmul(qh, k.transpose(-1, -2)) * (d ** -0.5)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    y = torch.matmul(probs.float(), v.float()).to(q.dtype).view(batch, nq * d)
    if out is not None:
        out.copy_(y)
        return out
    return y


def decode_attention_fused(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache,
                           eps, nq, nkv, out=None):
    """The fused path's twin: head norm, RoPE, cache write, then attention."""
    batch = qkv.shape[0]
    d = q_weight.shape[0]
    p = int(pos.item())
    parts = qkv.view(batch, -1, d)
    q = rms_norm(parts[:, :nq], q_weight, eps)
    k = rms_norm(parts[:, nq:nq + nkv], k_weight, eps)
    v = parts[:, nq + nkv:]
    c = cos[p].view(1, 1, d)
    s = sin[p].view(1, 1, d)
    k_cache[:, :, p] = _rope(k, c, s)
    v_cache[:, :, p] = v
    return decode_attention(_rope(q, c, s).reshape(batch, nq * d), k_cache, v_cache,
                            pos, nq, nkv, out=out)


def project(x, w, y, m, n, k, tile, ssq_in, ssq_out, eps, *, norm_w=None,
            glu=False, res=None, n_parts=1, ss_stride=1, part=None, ctr=None):
    """Signature-compatible twin of :func:`kernels.gemm.project`.

    The FP32 matmul is what the tile kernel's FP32 accumulator computes; only
    the order of the sum differs.  ``allow_tf32`` must be off, which the
    engine sets globally -- with it on, this twin is the inaccurate one.
    """
    dt = x.dtype
    rows = x[:m, :k]
    if norm_w is not None:
        ss = ssq_in[:n_parts, :m].float().sum(0)
        rstd = torch.rsqrt(ss / k + eps).view(m, 1)
        rows = (norm_w.float() * (rows.float() * rstd).to(dt).float()).to(dt)
    acc = torch.matmul(rows.float(), w[:n, :k].float().t())
    if glu:
        up = torch.matmul(rows.float(), w[n:2 * n, :k].float().t())
        g = acc.to(dt).float()
        u = up.to(dt).float()
        act = (g * torch.sigmoid(g)).to(dt).float()
        y[:m, :n] = (act * u).to(dt)
    elif res is not None:
        h = (res[:m, :n].float() + acc.to(dt).float()).to(dt)
        res[:m, :n] = h
        tiles = -(-n // tile.bn)
        hf = h.float().pow(2).view(m, tiles, tile.bn).sum(-1)
        ssq_out[:tiles, :m] = hf.t()
    else:
        y[:m, :n] = acc.to(dt)


def decode_attention_masked(q, k_cache, v_cache, pos, nq, nkv, out=None):
    """Exact GQA over the whole cache, masked by a device-side length.

    The twins above read ``pos`` on the host, which is fine for a load-time
    check on a 48-slot cache and fatal inside a CUDA graph.  This one indexes
    nothing on the host and expands nothing: the query is reshaped to
    ``[B, kv heads, group, D]`` so each KV head is read once, which is the
    same mapping the Triton kernel applies.  It is the step's floor -- what
    runs, and is still capturable, if the attention kernel is rejected.
    """
    batch = q.shape[0]
    d = k_cache.shape[-1]
    capacity = k_cache.shape[2]
    group = nq // nkv
    qh = q.view(batch, nkv, group, d)
    scores = torch.matmul(qh, k_cache.transpose(-1, -2)).float() * (d ** -0.5)
    valid = torch.arange(capacity, device=q.device) <= pos
    scores = scores.masked_fill(~valid.view(1, 1, 1, capacity), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v_cache.dtype)
    y = torch.matmul(probs, v_cache).reshape(batch, nq * d)
    if out is not None:
        out.copy_(y)
        return out
    return y


def decode_fused_masked(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache,
                        eps, nq, nkv, out=None):
    """:func:`decode_attention_masked` with the head norm, RoPE and cache
    write in front of it, all indexed on the device."""
    batch = qkv.shape[0]
    d = q_weight.shape[0]
    idx = pos.view(1)
    parts = qkv.view(batch, -1, d)
    q = rms_norm(parts[:, :nq], q_weight, eps)
    k = rms_norm(parts[:, nq:nq + nkv], k_weight, eps)
    v = parts[:, nq + nkv:]
    c = torch.index_select(cos, 0, idx).view(1, 1, d)
    s = torch.index_select(sin, 0, idx).view(1, 1, d)
    k_cache.index_copy_(2, idx, _rope(k, c, s).unsqueeze(2))
    v_cache.index_copy_(2, idx, v.unsqueeze(2))
    return decode_attention_masked(_rope(q, c, s).reshape(batch, nq * d),
                                   k_cache, v_cache, pos, nq, nkv, out=out)
