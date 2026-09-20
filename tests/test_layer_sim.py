"""Run the one-launch layer kernel on numpy and compare it with plain torch.

``engine/kernels/layer.py`` is the one kernel in the engine that cannot be
checked by the load-time twins, because by the time it runs there is no
cheaper thing to compare it against on the device. So check it here instead:
``tests/tlsim.py`` executes the kernel's own source over numpy buffers, and
the other side of the comparison is the layer written as eight lines of
tensor algebra in torch bfloat16. They share no code.

With a grid of one block every barrier is satisfied by its own arrival, so
this covers everything except the multi-block barrier itself: index
arithmetic, masks, the stage wiring, the sum-of-squares hand-off, the KV
write and the BF16 cast boundary.

    python tests/test_layer_sim.py

Needs numpy and torch. Triton is not needed and not installed here: the
kernel is imported against a shim whose ``language`` is the simulator.
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover
    print(f"skipped: {exc}")
    raise SystemExit(0)

import tlsim  # noqa: E402


class _Launcher:
    """Stands in for a Triton kernel handle: carries ``.fn`` for the shim."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            raise RuntimeError("no GPU here; drive the kernel through tlsim.run")
        return _launch


def _install_triton_shim():
    if "triton" in sys.modules:
        return
    fake = types.ModuleType("triton")

    def jit(fn=None, **kwargs):
        def wrap(f):
            return _Launcher(f)
        return wrap(fn) if fn is not None else wrap

    fake.jit = jit
    fake.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length() if n > 1 else 1
    fake.cdiv = lambda a, b: -(-a // b)
    fake.language = tlsim
    sys.modules["triton"] = fake
    sys.modules["triton.language"] = tlsim


_install_triton_shim()

from kernels import layer as klayer  # noqa: E402

BF = torch.bfloat16
EPS = 1e-6
FAILURES = []


def check(name, got, want, tol=0.02):
    got = torch.as_tensor(np.asarray(got, dtype=np.float32))
    want = want.float()
    err = (got - want).abs().max().item()
    scale = max(want.abs().max().item(), 1e-6)
    ok = err <= tol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={err:.4g} scale={scale:.4g}")
    if not ok:
        FAILURES.append(name)


def bf(*shape, scale=1.0, gen=None):
    return (torch.randn(*shape, generator=gen) * scale).to(BF)


def flat(t):
    return t.float().contiguous().view(-1).numpy().astype(np.float32)


def ptr(buf, name, dtype=tlsim.bfloat16):
    return tlsim.pointer(buf, dtype, name)


def rms(x, w):
    """Qwen3RMSNorm: reduce in FP32, cast to BF16, then multiply by weight."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w * (xf * torch.rsqrt(var + EPS)).to(x.dtype)).to(BF)


def mm(x, w):
    """BF16 operands, FP32 accumulate, one rounding at the end."""
    return (x.float() @ w.float().T).to(BF)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference(h, ws, kc, vc, pos, cos, sin, dims):
    """The layer, in tensor algebra, with the reference's rounding points."""
    nq, nkv, d = dims["nq"], dims["nkv"], dims["d"]
    m = h.shape[0]
    n = rms(h, ws["in_w"])
    qkv = mm(n, ws["qkv_w"])
    q = qkv[:, : nq * d].view(m, nq, d)
    k = qkv[:, nq * d: (nq + nkv) * d].view(m, nkv, d)
    v = qkv[:, (nq + nkv) * d:].view(m, nkv, d)
    q = rms(q, ws["qn_w"])
    k = rms(k, ws["kn_w"])
    c, s = cos[pos].view(1, 1, d), sin[pos].view(1, 1, d)
    q = ((q.float() * c.float()).to(BF).float() + (rotate_half(q).float() * s.float()).to(BF).float()).to(BF)
    k = ((k.float() * c.float()).to(BF).float() + (rotate_half(k).float() * s.float()).to(BF).float()).to(BF)
    kc = kc.clone()
    vc = vc.clone()
    kc[:, :, pos, :] = k
    vc[:, :, pos, :] = v
    # attention over slots [0, pos], FP32 softmax, probabilities in BF16
    group = nq // nkv
    out = torch.empty((m, nq, d), dtype=BF)
    for b in range(m):
        for head in range(nq):
            kv = head // group
            kk = kc[b, kv, : pos + 1, :].float()
            vv = vc[b, kv, : pos + 1, :].float()
            scores = (q[b, head].float() @ kk.T) / (d ** 0.5)
            p = torch.softmax(scores, dim=-1).to(BF).float()
            out[b, head] = (p @ vv).to(BF)
    attn = out.view(m, nq * d)
    o = mm(attn, ws["o_w"])
    h2 = (h.float() + o.float()).to(BF)
    mid = rms(h2, ws["post_w"])
    gu = mm(mid, ws["gu_w"])
    inter = ws["dn_w"].shape[1]
    g, u = gu[:, :inter], gu[:, inter:]
    act = ((torch.nn.functional.silu(g.float()).to(BF)).float() * u.float()).to(BF)
    dproj = mm(act, ws["dn_w"])
    h3 = (h2.float() + dproj.float()).to(BF)
    return h3, attn, act, kc, vc


def case(splits, block_n):
    gen = torch.Generator().manual_seed(7)
    nq, nkv, d = 4, 2, 16
    hid, inter = 64, 32
    n_qkv, k_o = (nq + 2 * nkv) * d, nq * d
    m = batch = 2
    cap, pos = 8, 5
    bm, bk = 16, 16
    bn_qkv, bn_o, bn_gu, bn_dn = 16, 8, 8, 8
    parts = 16
    grid = 1

    ws = {
        "in_w": bf(hid, scale=0.5, gen=gen) + 1,
        "post_w": bf(hid, scale=0.5, gen=gen) + 1,
        "qn_w": bf(d, scale=0.5, gen=gen) + 1,
        "kn_w": bf(d, scale=0.5, gen=gen) + 1,
        "qkv_w": bf(n_qkv, hid, scale=0.1, gen=gen),
        "o_w": bf(hid, k_o, scale=0.1, gen=gen),
        "gu_w": bf(2 * inter, hid, scale=0.1, gen=gen),
        "dn_w": bf(hid, inter, scale=0.1, gen=gen),
    }
    h = bf(m, hid, scale=1.0, gen=gen)
    kc = bf(batch, nkv, cap, d, scale=1.0, gen=gen)
    vc = bf(batch, nkv, cap, d, scale=1.0, gen=gen)
    angles = torch.arange(cap).float()[:, None] * torch.linspace(1.0, 0.1, d)[None, :]
    cos, sin = angles.cos().to(BF), angles.sin().to(BF)

    chunk = -(-cap // splits)
    chunk = -(-chunk // block_n) * block_n
    print(f"layer kernel on the simulator: one block, {splits} attention"
          f" split(s), chunk {chunk}")
    want_h, want_attn, want_act, want_kc, want_vc = reference(
        h, ws, kc, vc, pos, cos, sin, {"nq": nq, "nkv": nkv, "d": d})

    # buffers for the kernel
    hb = flat(h)
    qkv_b = np.zeros(m * n_qkv, dtype=np.float32)
    attn_b = np.zeros(m * k_o, dtype=np.float32)
    act_b = np.zeros(m * inter, dtype=np.float32)
    kb, vb = flat(kc), flat(vc)
    ssq_in = np.zeros((parts, bm), dtype=np.float32)
    ssq_in[0, :m] = h.float().pow(2).sum(-1).numpy()
    ssq_a = np.zeros((parts, bm), dtype=np.float32)
    ssq_b = np.zeros((parts, bm), dtype=np.float32)
    flags = np.zeros((6, klayer.FLAG_BLOCK), dtype=np.int32)
    po = np.zeros(batch * nq * splits * d, dtype=np.float32)
    pm = np.zeros(batch * nq * splits, dtype=np.float32)
    pl = np.zeros(batch * nq * splits, dtype=np.float32)
    posb = np.array([pos], dtype=np.int32)

    args = (
        ptr(hb, "h"), ptr(ssq_in.ravel(), "ssq_in", tlsim.float32), tlsim.T(np.int32(1)),
        ptr(flat(ws["in_w"]), "in_w"), ptr(flat(ws["post_w"]), "post_w"),
        ptr(flat(ws["qkv_w"]), "qkv_w"), ptr(flat(ws["qn_w"]), "qn_w"),
        ptr(flat(ws["kn_w"]), "kn_w"), ptr(flat(ws["o_w"]), "o_w"),
        ptr(flat(ws["gu_w"]), "gu_w"), ptr(flat(ws["dn_w"]), "dn_w"),
        ptr(qkv_b, "qkv"), ptr(attn_b, "attn"), ptr(act_b, "act"),
        ptr(kb, "k"), ptr(vb, "v"), ptr(posb, "pos", tlsim.int32),
        ptr(flat(cos), "cos"), ptr(flat(sin), "sin"),
        ptr(po, "po", tlsim.float32), ptr(pm, "pm", tlsim.float32),
        ptr(pl, "pl", tlsim.float32),
        ptr(ssq_a.ravel(), "ssq_a", tlsim.float32), ptr(ssq_b.ravel(), "ssq_b", tlsim.float32),
        ptr(flags.ravel(), "flags", tlsim.int32), tlsim.T(np.int32(0)),
        tlsim.T(np.int32(m)), tlsim.T(np.float32(EPS)),
        tlsim.T(np.float32((1.0 / d ** 0.5) * 1.4426950408889634)),
        tlsim.T(np.int32(nkv * cap * d)), tlsim.T(np.int32(cap * d)), tlsim.T(np.int32(d)),
        tlsim.T(np.int32(chunk)), tlsim.T(np.int32(splits)),
    )
    kwargs = dict(
        HID=hid, N_QKV=n_qkv, K_O=k_o, INTER=inter,
        NQ=nq, NKV=nkv, GROUP=nq // nkv, D=d,
        BATCH=batch, SPLITS=splits, SPLITS_P=max(2, 1 << (splits - 1).bit_length()),
        BM=bm, BK=bk,
        BN_QKV=bn_qkv, BN_O=bn_o, BN_GU=bn_gu, BN_DN=bn_dn,
        BLOCK_H=16, BLOCK_N=block_n, PARTS=parts,
        G=grid, SS_STRIDE=bm,
    )
    tlsim.run(klayer, "_layer_kernel", (grid,), args, kwargs,
              jit_names=("_arrive_and_wait",))

    check("residual stream h", hb.reshape(m, hid), want_h)
    check("attention output", attn_b.reshape(m, k_o), want_attn)
    check("SwiGLU activation", act_b.reshape(m, inter), want_act)
    check("K cache", kb.reshape(batch, nkv, cap, d), want_kc)
    check("V cache", vb.reshape(batch, nkv, cap, d), want_vc)
    # the sums of squares the next norm consumes, one partial per tile
    got_a = ssq_a[: hid // bn_o, :m].sum(axis=0)
    want_a = (want_h.float() - torch.as_tensor(np.asarray(act_b.reshape(m, inter)))
              .float() @ ws["dn_w"].float().T)
    check("sum of squares after O", got_a,
          want_a.pow(2).sum(-1), tol=0.05)
    got_b = ssq_b[: hid // bn_dn, :m].sum(axis=0)
    check("sum of squares after down", got_b, want_h.float().pow(2).sum(-1), tol=0.05)
    print(f"barrier flags set: {int(flags[:, :grid].sum())} of {6 * grid}")

    print(f"barriers fired: {int(flags[:, :grid].sum())}")


def main():
    case(1, 8)   # one split covers the sequence: reduce stage skipped
    case(2, 4)   # split attention: partials plus the reduce stage
    if FAILURES:
        print(f"FAILED: {', '.join(sorted(set(FAILURES)))}")
        return 1
    print("all layer-kernel checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
