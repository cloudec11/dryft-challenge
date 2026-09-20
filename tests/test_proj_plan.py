"""Checks for the parts of the engine that run without a GPU.

There is no CUDA here, so these cover the pure-Python decisions that a bad
choice in would only show up as a failed platform run: which tile shapes the
tuner is allowed to try, and whether the sum-of-squares hand-off actually
reconstructs an RMSNorm.

``triton`` is not installed locally either, so it is stubbed far enough for
``kernels.proj`` to import; nothing here launches a kernel.

    python -m pytest tests/test_proj_plan.py
"""

import os
import sys
import types

import pytest

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine")
sys.path.insert(0, ENGINE)


def _stub_triton():
    if "triton" in sys.modules:
        return
    tl = types.ModuleType("triton.language")
    tl.constexpr = object

    def _missing(*_a, **_k):  # pragma: no cover - never called here
        raise RuntimeError("stubbed triton")

    for name in ("arange", "zeros", "load", "store", "sum", "dot", "trans", "where",
                 "program_id", "full", "maximum", "max", "min", "exp", "static_range",
                 "cdiv", "num_programs"):
        setattr(tl, name, _missing)
    tl.math = types.SimpleNamespace(rsqrt=_missing, exp2=_missing)
    tl.float32 = "fp32"
    tl.int64 = "i64"

    triton = types.ModuleType("triton")
    triton.language = tl
    triton.jit = lambda fn: fn
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (n - 1).bit_length()
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = tl


_stub_triton()

from kernels import proj  # noqa: E402


SM = 132
HIDDEN, INTER, VOCAB = 2560, 9728, 151936
# (role, n, k, glu, res) for Qwen3-4B, as qwen.Weights.role_dims reports them.
ROLES = [
    ("qkv", 6144, HIDDEN, False, False),
    ("o", HIDDEN, 4096, False, True),
    ("gate_up", INTER, HIDDEN, True, False),
    ("down", HIDDEN, INTER, False, True),
    ("lm", VOCAB, HIDDEN, False, False),
]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def _block_m(m):
    return max(16, 1 << (m - 1).bit_length()) if m > 1 else 16


@pytest.mark.parametrize("m", BATCHES)
@pytest.mark.parametrize("role,n,k,glu,res", ROLES)
def test_candidates_are_launchable(m, role, n, k, glu, res):
    """Every shortlisted tile must divide the matrix and fit on an SM.

    A tile that does not is not merely slow: ``N // BLOCK_N`` silently drops
    output columns, and a K slice that does not divide evenly drops part of
    the sum.
    """
    cands = proj.candidates(m, n, k, glu, SM, prefer_wide=res)
    assert cands, f"{role} at batch {m} has no candidate tile"
    bm = _block_m(m)
    for cfg in cands:
        assert n % cfg.bn == 0, f"{role} {cfg}: {cfg.bn} does not divide n={n}"
        assert k % (cfg.bk * cfg.split) == 0, f"{role} {cfg}: bad K slice"
        assert cfg.split <= proj.MAX_SPLIT
        if not cfg.vec:
            assert cfg.bn >= 16 and cfg.bk >= 16, f"{role} {cfg}: tl.dot needs 16"
            assert proj.smem_bytes(cfg, bm, glu) <= proj.SMEM_MAX_BLOCK


@pytest.mark.parametrize("role,n,k,glu,res", ROLES)
def test_split_only_where_the_grid_is_too_small(role, n, k, glu, res):
    """Split-K costs a reduce launch, so it is offered only for the two
    projections whose 2560 columns cannot fill the device on their own."""
    splits = {c.split for c in proj.candidates(16, n, k, glu, SM, prefer_wide=res)}
    if n == HIDDEN:
        assert splits - {1}, f"{role} should be allowed to split K"
    else:
        assert splits == {1}, f"{role} should not split K"


@pytest.mark.parametrize("m", BATCHES)
@pytest.mark.parametrize("role,n,k,glu,res", ROLES)
def test_partials_fit_the_handoff_buffer(m, role, n, k, glu, res):
    """A producing role's program count becomes the number of sums of squares
    a consumer reads, and the buffer is fixed at load time."""
    if not res:
        return
    for cfg in proj.candidates(m, n, k, glu, SM, prefer_wide=True):
        parts = proj.n_tiles(cfg, n)
        block = max(2, 1 << (parts - 1).bit_length())
        assert block <= proj.SSQ_PARTS, f"{role} {cfg}: {parts} partials overflow"


@pytest.mark.parametrize("role,n,k,glu,res", ROLES)
def test_incumbent_leads_the_shortlist(role, n, k, glu, res):
    """The tuner only moves off its first candidate for a clear win, so the
    first candidate has to be the shape that already worked."""
    cands = proj.candidates(16, n, k, glu, SM, prefer_wide=res)
    assert (cands[0].bn, cands[0].bk, cands[0].split) == (16, 256, 1), repr(cands[0])


def test_wave_efficiency_is_the_first_sort_key():
    """The O projection is the case the scoring exists for: 2560 columns in
    tiles of 16 is 160 programs on 132 SMs, which runs as two rounds for 1.21
    rounds of work. Something better has to be offered above it."""
    cands = proj.candidates(16, HIDDEN, 4096, False, SM, prefer_wide=True)
    unsplit = next(c for c in cands if c.split == 1 and c.bn == 16)
    assert unsplit.ctas == 160
    assert unsplit.waves == pytest.approx(160 / 132 / 2, rel=1e-3)
    best = max(cands[1:], key=lambda c: c.waves)
    assert best.waves > 0.9, f"nothing balances the O projection: {best!r}"


@pytest.mark.parametrize("m", [1, 4, 16])
def test_batch_one_gets_the_single_row_kernel(m):
    """At one row the tile kernel pads x to 16 rows for tl.dot; the vector
    kernel exists so batch 1 does not pay that."""
    cands = proj.candidates(m, HIDDEN, 4096, False, SM)
    assert any(c.vec for c in cands) == (m == 1)


def test_reference_projection_reconstructs_rms_norm():
    """The whole sum-of-squares hand-off rests on one identity: partials
    summed over the producing programs equal ``sum(h ** 2)`` over the row, so
    the consumer's prologue is exactly Qwen3RMSNorm. If the count or the
    divisor were wrong this is where it shows."""
    torch = pytest.importorskip("torch")
    from kernels import ref

    gen = torch.Generator().manual_seed(7)
    m, k, n, parts = 3, 64, 32, 4
    eps = 1e-6
    h = torch.randn(m, k, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(k, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    # What a producing epilogue would have left behind: one partial per
    # program, each covering a slice of the row.
    ssq = torch.zeros(parts, 16, dtype=torch.float32)
    for p, chunk in enumerate(h.to(torch.float32).chunk(parts, dim=1)):
        ssq[p, :m] = chunk.pow(2).sum(-1)

    y = torch.zeros(m, n, dtype=torch.bfloat16)
    ref.project(h, w, y, m, n, k, ssq, ssq.clone(), eps, norm_w=weight, n_parts=parts)

    want = torch.nn.functional.linear(ref._rms(h, weight, eps), w)
    assert torch.equal(y, want)
