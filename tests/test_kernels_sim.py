"""Run the engine's Triton kernels on numpy and compare them with their twins.

The kernels ship as the only thing in this repository that cannot be checked
without an H100 -- so this checks the part of them that is not about
hardware: index arithmetic, masks, epilogues, the sum-of-squares hand-off and
the BF16 cast boundary.  ``tests/tlsim.py`` executes the kernel's own source
over numpy buffers, one program at a time, and every masked load and store
asserts that what it touches is inside its buffer.

The twins on the other side are ``kernels/reference.py`` in real PyTorch
bfloat16.  Agreement between the two is worth having because they share no
code: one is a tiled kernel indexing raw offsets, the other is four lines of
tensor algebra.

    python tests/test_kernels_sim.py

Needs numpy, and PyTorch for the twins.  Skips itself if either is missing.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    import numpy as np
    import torch
    import triton  # noqa: F401  (the kernels import it at module scope)
except ImportError as exc:  # pragma: no cover
    print(f"skipped: {exc}")
    raise SystemExit(0)

import tlsim  # noqa: E402
import planner  # noqa: E402
from kernels import attention as kattn, elementwise as kelem, gemm as kgemm  # noqa: E402
from kernels import reference as kref  # noqa: E402

BF16 = torch.bfloat16
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
    return condition


def close(got, want, label, tol=None):
    """BF16 values from two different summation orders: equal to within the
    last bit or two of the format, not bit-identical."""
    g = np.asarray(got, dtype=np.float32).ravel()
    w = np.asarray(want, dtype=np.float32).ravel()
    if g.shape != w.shape:
        return check(False, f"{label}: shape {g.shape} against {w.shape}")
    if not np.isfinite(g).all():
        return check(False, f"{label}: not finite")
    scale = float(np.abs(w).max()) or 1.0
    worst = float(np.abs(g - w).max())
    limit = tol if tol is not None else 0.02 * scale
    return check(worst <= limit, f"{label}: max difference {worst:.5f} "
                                 f"(limit {limit:.5f}, scale {scale:.3f})")


def rand(*shape, scale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return (torch.randn(*shape) * scale).to(BF16)


def flat(t):
    """A BF16 torch tensor as the float32 numpy buffer the shim works on."""
    return t.float().contiguous().numpy().ravel().copy()


def ptr(buf, name, dtype=tlsim.bfloat16):
    return tlsim.pointer(buf, dtype, name)


def i32(x):
    return tlsim.T(np.int32(x))


# ---------------------------------------------------------------------------
# the projection kernel
# ---------------------------------------------------------------------------


def run_proj(tile, m, n, k, *, norm, glu, res, parts=3, seed=0):
    """One projection through both implementations; returns (shim, twin)."""
    torch.manual_seed(seed)
    x = rand(m, k, scale=0.1)
    w = rand(2 * n, k, scale=0.05)
    nw = (torch.rand(k) + 0.5).to(BF16)
    ssq_in = (torch.rand(planner.MAX_SPLIT, 16) + 0.5).float()
    res_t = rand(m, n) if res else None
    eps = 1e-6
    ss_stride = 16
    tiles_m = -(-m // tile.bm)
    part_stride = tile.bm * tiles_m * n

    # --- the twin
    y_ref = torch.zeros(m, n, dtype=BF16)
    r_ref = res_t.clone() if res else None
    s_ref = torch.zeros(planner.MAX_SPLIT, 16)
    kref.project(x, w, y_ref, m, n, k, tile, ssq_in, s_ref, eps,
                 norm_w=nw if norm else None, glu=glu, res=r_ref,
                 n_parts=parts, ss_stride=ss_stride)

    # --- the kernel, on numpy
    xb, wb, nwb = flat(x), flat(w), flat(nw)
    yb = np.zeros(m * n, dtype=np.float32)
    rb = flat(res_t) if res else np.zeros(1, dtype=np.float32)
    pb = np.zeros(2 * planner.MAX_SPLIT * part_stride, dtype=np.float32)
    cb = np.zeros(max(1, tiles_m * (n // tile.bn)), dtype=np.int32)
    sib = ssq_in.numpy().ravel().copy()
    sob = np.zeros(planner.MAX_SPLIT * 16, dtype=np.float32)
    args = (ptr(xb, "x"), ptr(wb, "w"), ptr(yb, "y"), ptr(rb, "res"),
            ptr(pb, "part", tlsim.float32), ptr(cb, "ctr", tlsim.int32),
            ptr(sib, "ssq_in", tlsim.float32), ptr(sob, "ssq_out", tlsim.float32),
            ptr(nwb, "norm_w"), i32(m), i32(n), i32(k), i32(part_stride),
            i32(parts), eps)
    kwargs = dict(NORM=norm, GLU=glu, RES=res, SPLIT=tile.split,
                  FIXUP=tile.fixup, BLOCK_M=tile.bm, BLOCK_N=tile.bn,
                  BLOCK_K=tile.bk, SS_STRIDE=ss_stride)
    tlsim.run(kgemm, "_proj_kernel", (n // tile.bn, tiles_m, tile.split),
              args, kwargs, jit_names=("_epilogue", "_rstd_from_parts"))
    if tile.split > 1 and not tile.fixup:
        tlsim.run(kgemm, "_proj_reduce_kernel", (n // tile.bn, tiles_m),
                  (ptr(pb, "part", tlsim.float32), ptr(yb, "y"), ptr(rb, "res"),
                   ptr(sob, "ssq_out", tlsim.float32), i32(m), i32(n),
                   i32(part_stride)),
                  dict(GLU=glu, RES=res, SPLIT=tile.split, BLOCK_M=tile.bm,
                       BLOCK_N=tile.bn, SS_STRIDE=ss_stride),
                  jit_names=("_epilogue",))
    return (rb if res else yb), (r_ref if res else y_ref), sob, s_ref, cb


def test_projection_every_mode():
    m, n, k = 5, 64, 32
    tiles = [planner.Tile(16, 16, 16), planner.Tile(16, 32, 16),
             planner.Tile(16, 64, 32)]
    for tile in tiles:
        for norm, glu, res in ((True, False, False), (True, True, False),
                               (False, False, True), (False, False, False)):
            got, want, _, _, _ = run_proj(tile, m, n, k, norm=norm, glu=glu, res=res)
            close(got, want.float().numpy().ravel(),
                  f"proj {tile} norm={norm} glu={glu} res={res}")


def test_projection_hand_off():
    """A residual epilogue leaves one sum of squares per column tile, and the
    next RMSNorm has to be able to add them back up."""
    m, n, k = 4, 64, 32
    for tile in (planner.Tile(16, 16, 16), planner.Tile(16, 32, 32)):
        got, want, ssq, ssq_ref, _ = run_proj(tile, m, n, k, norm=False, glu=False,
                                              res=True)
        tiles_n = n // tile.bn
        close(ssq.reshape(planner.MAX_SPLIT, 16)[:tiles_n, :m],
              ssq_ref.numpy()[:tiles_n, :m], f"hand-off {tile}")
        total = ssq.reshape(planner.MAX_SPLIT, 16)[:tiles_n, :m].sum(0)
        rows = np.asarray(got, dtype=np.float32).reshape(m, n)
        close(total, (rows.astype(np.float64) ** 2).sum(-1), f"hand-off sum {tile}",
              tol=0.05 * float((rows ** 2).sum(-1).max()))


def test_split_k_matches_unsplit():
    """Splitting K is a reordering of one sum, so both paths, and both ways of
    reducing them, have to land in the same place."""
    m, n, k = 5, 64, 64
    base, base_ref, _, _, _ = run_proj(planner.Tile(16, 32, 16), m, n, k,
                                       norm=True, glu=False, res=False)
    for split in (2, 4):
        for fixup in (True, False):
            tile = planner.Tile(16, 32, 16, split, fixup=fixup)
            got, want, _, _, ctr = run_proj(tile, m, n, k, norm=True, glu=False,
                                            res=False)
            close(got, want.float().numpy().ravel(), f"split {split} fixup={fixup}")
            close(got, base, f"split {split} fixup={fixup} against unsplit")
            check(int(np.abs(ctr).sum()) == 0,
                  f"split {split} fixup={fixup}: counters left at {ctr.tolist()}")


def test_split_k_with_every_epilogue():
    m, n, k = 4, 64, 64
    for norm, glu, res in ((True, True, False), (False, False, True)):
        for fixup in (True, False):
            tile = planner.Tile(16, 32, 16, 2, fixup=fixup)
            got, want, _, _, _ = run_proj(tile, m, n, k, norm=norm, glu=glu, res=res)
            close(got, want.float().numpy().ravel(),
                  f"split epilogue norm={norm} glu={glu} res={res} fixup={fixup}")


def test_vector_kernel():
    """The batch-1 kernel is a different loop with the same arithmetic."""
    m, n, k = 1, 64, 32
    parts, ss_stride, eps = 3, 16, 1e-6
    for bn, bk in ((16, 16), (32, 32), (64, 16)):
        tile = planner.Tile(1, bn, bk, kind="vec")
        for norm, glu, res in ((True, False, False), (True, True, False),
                               (False, False, True), (False, False, False)):
            torch.manual_seed(5)
            x = rand(m, k, scale=0.1)
            w = rand(2 * n, k, scale=0.05)
            nw = (torch.rand(k) + 0.5).to(BF16)
            ssq_in = (torch.rand(planner.MAX_SPLIT, 16) + 0.5).float()
            res_t = rand(m, n) if res else None
            y_ref = torch.zeros(m, n, dtype=BF16)
            r_ref = res_t.clone() if res else None
            s_ref = torch.zeros(planner.MAX_SPLIT, 16)
            kref.project(x, w, y_ref, m, n, k, tile, ssq_in, s_ref, eps,
                         norm_w=nw if norm else None, glu=glu, res=r_ref,
                         n_parts=parts, ss_stride=ss_stride)
            yb = np.zeros(m * n, dtype=np.float32)
            rb = flat(res_t) if res else np.zeros(1, dtype=np.float32)
            sob = np.zeros(planner.MAX_SPLIT * 16, dtype=np.float32)
            tlsim.run(kgemm, "_proj_vec_kernel", (n // bn,),
                      (ptr(flat(x), "x"), ptr(flat(w), "w"), ptr(yb, "y"),
                       ptr(rb, "res"), ptr(ssq_in.numpy().ravel().copy(), "ssq_in",
                                           tlsim.float32),
                       ptr(sob, "ssq_out", tlsim.float32), ptr(flat(nw), "norm_w"),
                       i32(n), i32(k), i32(parts), eps),
                      dict(NORM=norm, GLU=glu, RES=res, BLOCK_N=bn, BLOCK_K=bk,
                           SS_STRIDE=ss_stride))
            got = rb if res else yb
            want = (r_ref if res else y_ref).float().numpy().ravel()
            close(got, want, f"vec {tile} norm={norm} glu={glu} res={res}")


def test_rms_norm_cast_boundary():
    """The engine's whole numerics budget in one test.

    Qwen3RMSNorm rounds to BF16 *between* the normalise and the weight.
    Rounding once at the end instead is more accurate and is a different
    function -- the contract calls that a reformulation and says it can move
    a logit by more than the noise floor.  So: the kernel must match the
    reference, and must not match the tidier version.
    """
    k = 256
    torch.manual_seed(2)
    x = rand(1, k, scale=1.0)
    nw = (torch.rand(k) * 2).to(BF16)
    eps = 1e-6
    ssq = torch.full((planner.MAX_SPLIT, 16), 0.0)
    ssq[0, 0] = float(x.float().pow(2).sum())
    tile = planner.Tile(16, 16, 32)
    yb = np.zeros(k, dtype=np.float32)
    weight = torch.eye(k).to(BF16)  # identity: the output is the normed row
    tlsim.run(kgemm, "_proj_kernel", (k // tile.bn, 1, 1),
              (ptr(flat(x), "x"), ptr(flat(weight), "w"), ptr(yb, "y"),
               ptr(np.zeros(1, dtype=np.float32), "res"),
               ptr(np.zeros(1, dtype=np.float32), "part", tlsim.float32),
               ptr(np.zeros(k, dtype=np.int32), "ctr", tlsim.int32),
               ptr(ssq.numpy().ravel().copy(), "ssq_in", tlsim.float32),
               ptr(np.zeros(planner.MAX_SPLIT * 16, dtype=np.float32), "ssq_out",
                   tlsim.float32),
               ptr(flat(nw), "norm_w"), i32(1), i32(k), i32(k), i32(0), i32(1), eps),
              dict(NORM=True, GLU=False, RES=False, SPLIT=1, FIXUP=False,
                   BLOCK_M=16, BLOCK_N=tile.bn, BLOCK_K=tile.bk, SS_STRIDE=16),
              jit_names=("_epilogue", "_rstd_from_parts"))
    reference = kref.rms_norm(x, nw, eps).float().numpy().ravel()
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    tidier = (x.float() * rstd * nw.float()).to(BF16).float().numpy().ravel()
    close(yb, reference, "rms norm against the reference", tol=2e-2)
    check(not np.allclose(reference, tidier, atol=0, rtol=0),
          "this test cannot tell the two formulations apart; it proves nothing")


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def run_attention(batch, nq, nkv, d, capacity, pos, splits, fixup, seed=0):
    torch.manual_seed(seed)
    group = nq // nkv
    block_h = max(16, 1 << (group - 1).bit_length())
    chunk = -(-capacity // splits)
    splits = -(-capacity // chunk)
    qkv = rand(batch, (nq + 2 * nkv) * d, scale=0.5)
    qw = (torch.rand(d) + 0.5).to(BF16)
    kw = (torch.rand(d) + 0.5).to(BF16)
    cos = rand(capacity + 8, d)
    sin = rand(capacity + 8, d)
    k_cache = rand(batch, nkv, capacity, d, scale=0.3)
    v_cache = rand(batch, nkv, capacity, d, scale=0.3)
    pos_t = torch.tensor(pos, dtype=torch.int64)

    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    want = kref.decode_attention_fused(qkv, qw, kw, cos, sin, pos_t, k_ref, v_ref,
                                       1e-6, nq, nkv)

    kb, vb = flat(k_cache), flat(v_cache)
    out = np.zeros(batch * nq * d, dtype=np.float32)
    n_part = batch * nq * splits
    po = np.zeros(n_part * d, dtype=np.float32)
    pm = np.zeros(n_part, dtype=np.float32)
    pl = np.zeros(n_part, dtype=np.float32)
    ctr = np.zeros(batch * nkv, dtype=np.int32)
    posb = np.array([pos], dtype=np.int64)
    args = (ptr(flat(qkv), "qkv"), ptr(flat(qw), "qw"), ptr(flat(kw), "kw"),
            ptr(flat(cos), "cos"), ptr(flat(sin), "sin"),
            ptr(posb, "pos", tlsim.int64), ptr(kb, "k"), ptr(vb, "v"),
            ptr(po, "po", tlsim.float32), ptr(pm, "pm", tlsim.float32),
            ptr(pl, "pl", tlsim.float32), ptr(ctr, "ctr", tlsim.int32),
            ptr(out, "out"),
            i32(nkv * capacity * d), i32(capacity * d), i32(d), i32(chunk),
            i32(splits), (d ** -0.5) * math.log2(math.e), 1e-6)
    kwargs = dict(NQ=nq, NKV=nkv, GROUP=group, D=d, BLOCK_H=block_h, BLOCK_N=32,
                  SPLITS=1 << (splits - 1).bit_length(), ONE_SPLIT=splits == 1,
                  FIXUP=fixup and splits > 1, FUSE=True)
    tlsim.run(kattn, "_attn_kernel", (batch, nkv, splits), args, kwargs)
    if splits > 1 and not fixup:
        tlsim.run(kattn, "_attn_reduce_kernel", (batch, nq),
                  (ptr(po, "po", tlsim.float32), ptr(pm, "pm", tlsim.float32),
                   ptr(pl, "pl", tlsim.float32), ptr(out, "out"), i32(splits)),
                  dict(NQ=nq, D=d, SPLITS=1 << (splits - 1).bit_length()))
    return out, want, (kb, vb), (k_ref, v_ref), ctr, splits


def test_attention_against_the_twin():
    for splits, fixup in ((1, False), (2, False), (2, True), (3, True), (5, True)):
        out, want, caches, refs, ctr, used = run_attention(
            batch=2, nq=4, nkv=2, d=16, capacity=12, pos=7, splits=splits,
            fixup=fixup)
        close(out, want.float().numpy().ravel(),
              f"attention splits={used} fixup={fixup}")
        close(caches[0], refs[0].float().numpy().ravel(),
              f"K cache splits={used} fixup={fixup}", tol=0.0)
        close(caches[1], refs[1].float().numpy().ravel(),
              f"V cache splits={used} fixup={fixup}", tol=0.0)
        if fixup and used > 1:
            check(int(np.abs(ctr).sum()) == 0,
                  f"attention counters left at {ctr.tolist()}")


def test_attention_reads_exactly_the_written_slots():
    """The step's K and V replace slot ``pos`` in registers, and only the
    split that owns that slot stores them.  A split boundary landing on
    ``pos`` is where that goes wrong."""
    for pos in (0, 1, 3, 4, 7, 11):
        out, want, _, _, _, used = run_attention(batch=1, nq=4, nkv=1, d=16,
                                                 capacity=12, pos=pos, splits=3,
                                                 fixup=True, seed=pos)
        close(out, want.float().numpy().ravel(), f"attention at pos={pos}")


# ---------------------------------------------------------------------------
# the small kernels
# ---------------------------------------------------------------------------


def test_embed():
    vocab, hidden, m = 64, 128, 3
    torch.manual_seed(1)
    emb = rand(vocab, hidden)
    ids = torch.randint(0, vocab, (m,))
    h_ref = torch.zeros(m, hidden, dtype=BF16)
    s_ref = torch.zeros(4, m)
    kref.embed(ids, emb, h_ref, s_ref)
    hb = np.zeros(m * hidden, dtype=np.float32)
    sb = np.zeros(4 * m, dtype=np.float32)
    tlsim.run(kelem, "_embed_kernel", (m,),
              (ptr(ids.numpy().ravel().copy(), "ids", tlsim.int64),
               ptr(flat(emb), "emb"), ptr(hb, "h"), ptr(sb, "ssq", tlsim.float32),
               i32(hidden)), dict(BLOCK=hidden))
    close(hb, h_ref.float().numpy().ravel(), "embed rows", tol=0.0)
    close(sb[:m], s_ref[0].numpy(), "embed sum of squares")


def test_add_rms_norm():
    rows, hidden = 3, 128
    for add in (True, False):
        torch.manual_seed(4)
        x = rand(rows, hidden)
        r = rand(rows, hidden) if add else None
        w = (torch.rand(hidden) + 0.5).to(BF16)
        r_ref = r.clone() if add else None
        want = kref.add_rms_norm(x, r_ref, w, 1e-6)
        xb = flat(x)
        rb = flat(r) if add else xb
        ob = np.zeros(rows * hidden, dtype=np.float32)
        tlsim.run(kelem, "_add_rms_norm_kernel", (rows,),
                  (ptr(xb, "x"), ptr(rb, "r"), ptr(flat(w), "w"), ptr(ob, "out"),
                   i32(hidden), 1e-6), dict(BLOCK=hidden, ADD=add))
        close(ob, want.float().numpy().ravel(), f"add_rms_norm add={add}")
        if add:
            close(rb, r_ref.float().numpy().ravel(), "add_rms_norm residual", tol=0.0)


def test_silu_mul():
    rows, inter = 3, 64
    torch.manual_seed(6)
    gu = rand(rows, 2 * inter)
    want = kref.silu_mul(gu)
    ob = np.zeros(rows * inter, dtype=np.float32)
    tlsim.run(kelem, "_silu_mul_kernel", (rows, 1),
              (ptr(flat(gu), "gu"), ptr(ob, "out"), i32(inter)), dict(BLOCK=64))
    close(ob, want.float().numpy().ravel(), "silu_mul")


def test_qkv_post_including_chunks():
    """Prefill in one pass and in two, which is where a chunk-relative row
    index would put a token at the wrong position or in the wrong sequence."""
    batch, T, nq, nkv, d, capacity = 2, 4, 4, 2, 16, 8
    rows = batch * T
    torch.manual_seed(9)
    qkv = rand(rows, (nq + 2 * nkv) * d, scale=0.5)
    qw = (torch.rand(d) + 0.5).to(BF16)
    kw = (torch.rand(d) + 0.5).to(BF16)
    cos = rand(capacity + 4, d)
    sin = rand(capacity + 4, d)
    pos = torch.zeros((), dtype=torch.int64)

    k_ref = torch.zeros(batch, nkv, capacity, d, dtype=BF16)
    v_ref = torch.zeros_like(k_ref)
    q_ref = kref.qkv_post(qkv, qw, kw, cos, sin, pos, k_ref, v_ref, T, nq, nkv, 1e-6)

    for chunk in (rows, T, 2 * T):
        kb = np.zeros(batch * nkv * capacity * d, dtype=np.float32)
        vb = np.zeros_like(kb)
        qb = np.zeros(rows * nq * d, dtype=np.float32)
        for lo in range(0, rows, chunk):
            hi = min(lo + chunk, rows)
            width = (nq + 2 * nkv) * d
            sub = flat(qkv)[lo * width:hi * width].copy()
            out = np.zeros((hi - lo) * nq * d, dtype=np.float32)
            tlsim.run(kelem, "_qkv_post_kernel",
                      (hi - lo, -(-(nq + 2 * nkv) // 16)),
                      (ptr(sub, "qkv"), ptr(flat(qw), "qw"), ptr(flat(kw), "kw"),
                       ptr(flat(cos), "cos"), ptr(flat(sin), "sin"),
                       ptr(np.zeros(1, dtype=np.int64), "pos", tlsim.int64),
                       ptr(out, "q"), ptr(kb, "k"), ptr(vb, "v"), i32(T), i32(lo),
                       i32(nkv * capacity * d), i32(capacity * d), i32(d), 1e-6),
                      dict(NQ=nq, NKV=nkv, D=d, BLOCK_HD=16))
            qb[lo * nq * d:hi * nq * d] = out
        close(qb, q_ref.float().numpy().ravel(), f"qkv_post q (chunk {chunk})")
        close(kb, k_ref.float().numpy().ravel(), f"qkv_post K (chunk {chunk})", tol=0.0)
        close(vb, v_ref.float().numpy().ravel(), f"qkv_post V (chunk {chunk})", tol=0.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        before = len(FAILURES)
        try:
            test()
            status = "ok" if len(FAILURES) == before else "FAIL"
        except Exception as exc:  # pragma: no cover
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
            status = "FAIL"
        print(f"{status:>4}  {test.__name__}", flush=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for message in FAILURES[:25]:
            print(f"  - {message}")
        return 1
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
