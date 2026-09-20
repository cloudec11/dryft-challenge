"""Checks for the part of the engine that decides things, not the part that
computes them.

``engine/planner.py`` has no torch and no triton in it on purpose: the whole
decode plan -- which tile shapes are legal, which one the traffic model
prefers, how the measured constants are fitted -- is arithmetic over integers
and can be checked on a machine with no GPU, which is where it gets written.

A bad decision here does not raise.  It produces a kernel that reads past its
partial buffer, or a tile the fused step then rejects, or a plan that is
quietly slower than the one v8 shipped.  So these are the invariants the
kernels depend on and cannot check for themselves.

    python tests/test_planner.py        # no pytest needed
    python -m pytest tests/test_planner.py
"""

import math
import os
import sys

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "engine")
sys.path.insert(0, ENGINE)

import planner  # noqa: E402

# Qwen3-4B, as the engine will ask for it: (name, n, k, glu).
ROLES = [
    ("qkv", 6144, 2560, False),
    ("o", 2560, 4096, False),
    ("gate_up", 9728, 2560, True),
    ("down", 2560, 9728, False),
    ("lm", 151936, 2560, False),
]
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 200)
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def test_tiles_divide_their_matrix():
    """Neither kernel masks its column or K index: a tile that does not
    divide the matrix would drop output columns silently."""
    dev = planner.Device()
    for batch in BATCHES:
        for name, n, k, glu in ROLES:
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           allow_split=name != "lm"):
                check(n % tile.bn == 0, f"{name} B={batch}: {tile} leaves {n % tile.bn}"
                                        f" of {n} columns uncovered")
                check(k % tile.bk == 0, f"{name} B={batch}: {tile} leaves"
                                        f" {k % tile.bk} of {k} uncovered")
                check(k % tile.split == 0, f"{name} B={batch}: {tile} splits {k}"
                                           " unevenly")
                check((k // tile.split) % tile.bk == 0,
                      f"{name} B={batch}: {tile} slice is not whole iterations")


def test_split_implies_a_reduction_plan():
    dev = planner.Device()
    for batch in BATCHES:
        for name, n, k, glu in ROLES:
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           allow_split=name != "lm"):
                check(tile.split == 1 or tile.split <= planner.MAX_SPLIT,
                      f"{tile} splits deeper than the partial buffer allows")
                check(not (tile.fixup and tile.split == 1),
                      f"{tile} claims a fixup with nothing to reduce")
                if name == "lm":
                    check(tile.split == 1, f"lm B={batch}: {tile} splits, but its"
                                           " partials are 78 MB a slice")


def test_fixup_is_not_offered_when_it_is_not_available():
    dev = planner.Device()
    for batch in (1, 16, 64):
        for name, n, k, glu in ROLES:
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           allow_fixup=False):
                check(not tile.fixup, f"{name}: {tile} uses a rejected fixup")


def test_partial_buffer_is_never_overrun():
    """The engine sizes one FP32 partial buffer for every role; a tile whose
    slices would not fit has to be refused here, not discovered by a kernel
    writing past the end of it."""
    dev = planner.Device()
    hidden, inter, qkv_n = 2560, 9728, 6144
    for batch in BATCHES:
        block_m = planner.block_m_for(batch)
        rows = math.ceil(batch / block_m) * block_m
        capacity = planner.MAX_SPLIT * rows * max(2 * inter, qkv_n, hidden)
        for name, n, k, glu in ROLES:
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           allow_split=name != "lm",
                                           partial_capacity=capacity):
                if tile.split == 1:
                    continue
                need = (tile.split * math.ceil(batch / tile.bm) * tile.bm * n
                        * (2 if glu else 1))
                check(need <= capacity,
                      f"{name} B={batch}: {tile} needs {need} floats of {capacity}")


def test_producers_fit_the_hand_off_buffer():
    """``o`` and ``down`` leave one partial sum of squares per column tile;
    the consumer reads them out of a fixed-height buffer."""
    dev = planner.Device()
    for batch in BATCHES:
        for name, n, k, glu in (("o", 2560, 4096, False), ("down", 2560, 9728, False)):
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           max_parts=256):
                check(math.ceil(n / tile.bn) <= 256,
                      f"{name} B={batch}: {tile} needs {math.ceil(n / tile.bn)}"
                      " hand-off rows")


def test_incumbent_is_offered_first():
    """The tuner starts from the tile every engine since v1 has run, so a
    model that is wrong about this machine costs nothing."""
    dev = planner.Device()
    for batch in (1, 4, 16, 64):
        for name, n, k, glu in ROLES:
            incumbent = planner.Tile(planner.block_m_for(batch), 16, 256)
            got = planner.candidates(batch, n, k, glu, dev, limit=5,
                                     incumbent=incumbent,
                                     allow_split=name != "lm")
            check(got and got[0].key == incumbent.key,
                  f"{name} B={batch}: shortlist starts with {got[0] if got else None}")


def test_shortlist_is_ordered_and_distinct():
    dev = planner.Device()
    for batch in (1, 16, 64):
        for name, n, k, glu in ROLES:
            got = planner.candidates(batch, n, k, glu, dev, limit=8,
                                     allow_split=name != "lm")
            check(len(got) == len({t.key for t in got}),
                  f"{name} B={batch}: shortlist repeats a tile")
            check(all(math.isfinite(t.score) for t in got),
                  f"{name} B={batch}: a candidate scored infinite")
            check(got == sorted(got, key=lambda t: t.score) or len(got) < 2
                  or got[0].score <= got[-1].score + 1e-12,
                  f"{name} B={batch}: shortlist is not ordered by model time")


def test_occupancy_respects_shared_memory():
    dev = planner.Device()
    for batch in (1, 16, 64):
        for name, n, k, glu in ROLES:
            for tile in planner.candidates(batch, n, k, glu, dev, limit=12,
                                           allow_split=name != "lm"):
                need = planner.smem_bytes(tile, glu)
                check(need <= dev.smem_per_block,
                      f"{name}: {tile} asks for {need} bytes of shared memory")
                check(planner.blocks_per_sm(tile, dev, glu) >= 1,
                      f"{name}: {tile} does not fit on an SM at all")


def test_wave_efficiency_is_a_fraction():
    for programs in (1, 5, 131, 132, 133, 264, 1000, 9496):
        for resident in (132, 264, 528):
            eff = planner.wave_efficiency(programs, resident)
            check(0.0 < eff <= 1.0, f"{programs}/{resident} -> {eff}")
    check(planner.wave_efficiency(264, 264) == 1.0, "a full wave is not 1.0")
    check(planner.wave_efficiency(0, 132) == 0.0, "an empty grid is not 0")


def test_wide_tiles_cut_the_activation_re_read():
    """The whole premise, stated as a test: at ``BLOCK_N = 16`` the x term is
    the same size as the weights, and widening the tile divides it."""
    dev = planner.Device()
    n, k = 6144, 2560
    weights, narrow_l2, _ = planner.traffic(planner.Tile(16, 16, 256), 16, n, k,
                                            False, dev)
    _, wide_l2, _ = planner.traffic(planner.Tile(16, 128, 64, 8), 16, n, k, False, dev)
    check(abs(narrow_l2 - weights) / weights < 0.05,
          f"x re-reads {narrow_l2} against {weights} of weights at BLOCK_N=16")
    check(wide_l2 < narrow_l2 / 2,
          f"a 128-wide tile re-reads {wide_l2}, against {narrow_l2}")


def test_split_does_not_add_activation_traffic():
    """Splitting K restores the program count without paying for it twice --
    which is the reason a wide tile is affordable at all."""
    dev = planner.Device()
    n, k = 2560, 4096
    base = planner.traffic(planner.Tile(16, 128, 64, 1), 16, n, k, False, dev)[1]
    for split in (2, 4, 8):
        got = planner.traffic(planner.Tile(16, 128, 64, split), 16, n, k, False, dev)[1]
        partials = 2 * split * 16 * n * 4
        check(abs(got - base - partials) < 1,
              f"split {split} changed the x term, not only the partials")


def test_calibration_recovers_planted_constants():
    """``fit_l2_cost`` is solved, not fitted, so feeding it two times the
    model itself produced has to give the constants back."""
    truth = planner.Device(bw=3.05e12, launch_s=2.4e-6, l2_cost=0.42)
    n, k, m = 6144, 2560, 16
    narrow = planner.Tile(16, 16, 256)
    wide = planner.Tile(16, 128, 64, 8, fixup=True)
    pair = [(t, planner.model_time(t, m, n, k, False, truth)) for t in (narrow, wide)]
    start = planner.Device(launch_s=truth.launch_s)  # launch is measured first
    bw, l2 = planner.fit_l2_cost(pair[0], pair[1], m, n, k, False, start)
    check(bw is not None and abs(bw - truth.bw) / truth.bw < 0.12,
          f"bandwidth came back {bw} for {truth.bw}")
    check(l2 is not None and abs(l2 - truth.l2_cost) < 0.2,
          f"l2 cost came back {l2} for {truth.l2_cost}")


def test_calibration_refuses_nonsense():
    dev = planner.Device()
    check(dev.calibrated(bw=-1.0).bw == dev.bw, "a negative bandwidth was kept")
    check(dev.calibrated(bw=float("nan")).bw == dev.bw, "a NaN bandwidth was kept")
    check(dev.calibrated(launch_s=1.0).launch_s == 30e-6,
          "a 1-second launch was not clamped")
    check(dev.calibrated(l2_cost=None).l2_cost == dev.l2_cost,
          "a missing l2 cost was not left alone")
    check(dev.calibrated(bw=3.3e12).bw == 3.3e12, "a plausible bandwidth was rejected")


def test_block_m_covers_the_batch_in_one_tile():
    for batch in BATCHES:
        bm = planner.block_m_for(batch)
        check(bm >= planner.DOT_MIN_M, f"B={batch}: BLOCK_M {bm} is below tl.dot's 16")
        check(bm & (bm - 1) == 0, f"B={batch}: BLOCK_M {bm} is not a power of two")
        check(bm >= batch or bm == planner.MAX_BLOCK_M,
              f"B={batch}: BLOCK_M {bm} tiles the rows without need")


def test_the_model_prefers_what_v12_diagnosed():
    """Not a prediction, an ordering check: whatever the measured constants
    turn out to be, the model must rank the wide-plus-split tile above the
    16-wide one at every batch the hidden set could use."""
    dev = planner.Device()
    for batch in (4, 16, 32):
        for name, n, k, glu in ROLES:
            best = planner.candidates(batch, n, k, glu, dev, limit=1,
                                      allow_split=name != "lm")[0]
            incumbent = planner.Tile(planner.block_m_for(batch), 16, 256)
            check(best.score <= planner.model_time(incumbent, batch, n, k, glu, dev),
                  f"{name} B={batch}: model ranks {incumbent} above {best}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        before = len(FAILURES)
        test()
        status = "ok" if len(FAILURES) == before else "FAIL"
        print(f"{status:>4}  {test.__name__}")
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for message in FAILURES[:40]:
            print(f"  - {message}")
        return 1
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
