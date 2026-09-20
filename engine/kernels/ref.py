"""PyTorch twins of every Triton kernel.

Two jobs. They are the reference each kernel is checked against at load time,
and they are the fallback if Triton cannot compile in the sandbox -- an
engine that cannot build a kernel should get slower, never wrong.

Each one is written the way ``modeling_qwen3`` writes it, so the comparison
is against the reference's own rounding rather than against a tidier version
of the same maths.
"""

import math

import torch
import torch.nn.functional as F


def add_rms_norm(x, residual, weight, eps):
    h = x if residual is None else residual + x
    hf = h.to(torch.float32)
    variance = hf.pow(2).mean(-1, keepdim=True)
    normed = (hf * torch.rsqrt(variance + eps)).to(h.dtype)
    return weight * normed, h


def _rms(x, weight, eps):
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    return weight * (xf * torch.rsqrt(variance + eps)).to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qkv_post(qkv, q_weight, k_weight, cos, sin, pos, k_cache, v_cache, T, nq, nkv, eps):
    m = qkv.shape[0]
    d = q_weight.shape[0]
    b = m // T
    x = qkv.view(b, T, nq + 2 * nkv, d)
    q = _rms(x[:, :, :nq], q_weight, eps)
    k = _rms(x[:, :, nq:nq + nkv], k_weight, eps)
    v = x[:, :, nq + nkv:]
    positions = pos.to(torch.int64) + torch.arange(T, device=qkv.device)
    c = cos.index_select(0, positions)[None, :, None, :]
    s = sin.index_select(0, positions)[None, :, None, :]
    q = (q * c) + (_rotate_half(q) * s)
    k = (k * c) + (_rotate_half(k) * s)
    k_cache.index_copy_(2, positions, k.transpose(1, 2))
    v_cache.index_copy_(2, positions, v.transpose(1, 2))
    return q.reshape(m, nq * d)


def silu_mul(gate_up):
    inter = gate_up.shape[1] // 2
    return F.silu(gate_up[:, :inter]) * gate_up[:, inter:]


def embed(ids, emb_w, h, ssq):
    x = F.embedding(ids, emb_w)
    h.copy_(x)
    ssq.zero_()
    ssq[0, :ids.shape[0]] = x.to(torch.float32).pow(2).sum(-1)


def decode_attention(q, k_cache, v_cache, pos, nq, nkv):
    """FP32 masked attention over the whole capacity: graph-safe, slow, and
    accurate enough to be the arbiter for the split kernel."""
    b, _, cap, d = k_cache.shape
    visible = torch.arange(cap, device=q.device) <= pos.to(torch.int64)
    qg = q.view(b, nkv, nq // nkv, d).to(torch.float32)
    scores = torch.matmul(qg, k_cache.to(torch.float32).transpose(-1, -2)) / math.sqrt(d)
    scores = scores.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_cache.to(torch.float32))
    return out.reshape(b, nq * d).to(q.dtype)


def project(x, w, y, m, n, k, ssq_in, ssq_out, eps, norm_w=None, n_parts=1,
            res=None, glu=False):
    """The twin of :func:`kernels.proj.project`, epilogues included."""
    xs = x[:m]
    if norm_w is not None:
        rstd = torch.rsqrt(ssq_in[:n_parts, :m].sum(0) / k + eps)
        xn = (xs.to(torch.float32) * rstd[:, None]).to(xs.dtype)
        xs = norm_w * xn
    acc = F.linear(xs, w[:n] if glu else w)
    if glu:
        up = F.linear(xs, w[n:])
        y[:m] = F.silu(acc) * up
    elif res is not None:
        h = (res[:m] + acc).to(res.dtype)
        res[:m] = h
        ssq_out.zero_()
        ssq_out[0, :m] = h.to(torch.float32).pow(2).sum(-1)
    else:
        y[:m] = acc
