"""The decode plan: measure the machine, then solve for the schedule.

Every engine in this repository before v13 chose its tiles the same way: a
hand-written shortlist per projection, compiled and raced at warmup, keep the
winner.  That method has two costs the run reports made visible.  Compiling
ten specialisations of five roles is most of the warmup budget, so the search
had to stay narrow; and a race decides between whatever happened to be on the
list, so the list *is* the search space.  v11 widened the list, spent the
budget, and came back with a step time identical to v8's.

This module inverts that.  The candidate space is generated from the device
and the role's own dimensions, every point in it is scored by a traffic model
whose constants are *measured on the machine at warmup*, and only the few
points the model cannot separate are compiled and raced.  Measurement still
decides; it just no longer has to do the searching as well.

The model
---------

A decode projection is ``y[m, n] = x[m, k] @ w[n, k].T`` with ``m`` equal to
the batch (1 to a few dozen) and ``n``, ``k`` in the thousands.  It is a
stream of one weight matrix past a handful of rows, so its time is the time
to move bytes, and there are four kinds:

``weights``   ``n * k * 2`` bytes from HBM, once per m-tile.  Irreducible.
``x``         each of the ``n / BLOCK_N`` column-tile programs reads the
              whole ``[m, k]`` input, so this term is ``n / BLOCK_N`` times
              the input.  It is served by L2 -- x is 80 KB at batch 16 -- but
              it competes for the same issue slots and L2 ports, so it is
              charged at ``l2_cost`` of an HBM byte, not zero.  At the
              ``BLOCK_N = 16`` tile that v1 through v11 ran, this term is
              6.25 GB per step against 8.05 GB of weights.
``partials``  ``split`` FP32 copies of the output, written and read back,
              when K is split.  Charged at the full HBM price, not the L2
              one: it is megabytes, and between the write and the read the
              same kernel streams up to a hundred megabytes of weights past
              the same cache.  x survives that; this does not.
``output``    ``m * n * 2`` bytes, negligible, counted anyway.

Widening ``BLOCK_N`` divides the x term, and it also divides the number of
programs -- which is why nobody could simply widen it.  Splitting K restores
the program count without adding any x traffic (each of the ``split``
programs for a column tile reads ``k / split`` of the row, and they sum to
the same ``k``).  So the two knobs have to be turned together, and that is
what :func:`candidates` enumerates.

The calibration
---------------

Three constants make the model concrete, and all three are measured in a few
hundred milliseconds at warmup (:mod:`engine` drives it, this module only
holds the arithmetic):

``bw``        bytes per second the device actually delivers on a stream of
              this shape.  Not the spec sheet: the number the machine gives
              on the day, which absorbs clock behaviour and the sandbox.
``launch_s``  seconds of wall clock a kernel launch costs inside a replayed
              graph.  This decides whether a split-K reduce is worth its
              second launch, and it is the single number that made the
              difference between v6 and v5.
``l2_cost``   the price of an L2 byte relative to an HBM byte.  Fitted from
              two measurements of the same projection at two ``BLOCK_N``
              values, which differ almost only in their x traffic.

Nothing here imports torch or triton, so the whole plan is testable on a
machine with no GPU -- which is where it gets written.
"""

import math

# ---------------------------------------------------------------------------
# Fallbacks.  Used only until calibration replaces them, and if calibration
# fails they are what the model runs on.  H100 SXM numbers.
# ---------------------------------------------------------------------------

DEFAULT_BW = 2.6e12          # bytes/s actually achieved by a streaming kernel
DEFAULT_LAUNCH_S = 2.8e-6    # per kernel inside a replayed graph, measured on v6
DEFAULT_L2_COST = 0.6        # an L2 byte against an HBM byte
DEFAULT_SMS = 132
SMEM_PER_SM = 227 * 1024     # H100 shared memory per SM
SMEM_PER_BLOCK = 200 * 1024  # the most one block may ask for before we skip it

# tl.dot needs 16 rows; below that the vector kernel is the alternative.
DOT_MIN_M = 16
# The largest BLOCK_M worth compiling.  Above it, weights are re-read once per
# m-tile and the projection is no longer purely a stream.
MAX_BLOCK_M = 128
# Hard ceiling on K slices, which bounds the FP32 partial buffer.
MAX_SPLIT = 16
# Rows in the sum-of-squares hand-off buffer: a producing role may not use a
# tile that needs more partials than this.
MAX_SSQ_PARTS = 768
# Registers make a wide FP32 accumulator expensive; keep the vector kernel's
# [BLOCK_N, BLOCK_K] accumulator inside this many elements.
VEC_ACC_MAX = 8192


class Device:
    """What the model needs to know about the machine.

    ``sms`` and the shared-memory limits come from the driver; ``bw``,
    ``launch_s`` and ``l2_cost`` start at the fallbacks above and are replaced
    by :meth:`calibrated` once the warmup microbenchmarks have run.
    """

    __slots__ = ("sms", "smem_per_sm", "smem_per_block", "bw", "launch_s", "l2_cost")

    def __init__(self, sms=DEFAULT_SMS, smem_per_sm=SMEM_PER_SM,
                 smem_per_block=SMEM_PER_BLOCK, bw=DEFAULT_BW,
                 launch_s=DEFAULT_LAUNCH_S, l2_cost=DEFAULT_L2_COST):
        self.sms = int(sms)
        self.smem_per_sm = int(smem_per_sm)
        self.smem_per_block = int(smem_per_block)
        self.bw = float(bw)
        self.launch_s = float(launch_s)
        self.l2_cost = float(l2_cost)

    def calibrated(self, bw=None, launch_s=None, l2_cost=None):
        """A copy with the measured constants substituted, each clamped to a
        range a plausible machine can produce.  A microbenchmark that lands
        outside it measured something other than what it meant to."""
        return Device(
            self.sms, self.smem_per_sm, self.smem_per_block,
            _clamp(bw, 0.3e12, 5.0e12, self.bw),
            _clamp(launch_s, 0.2e-6, 30e-6, self.launch_s),
            _clamp(l2_cost, 0.05, 1.0, self.l2_cost),
        )

    def __repr__(self):
        return (f"Device(sms={self.sms}, bw={self.bw / 1e12:.2f} TB/s, "
                f"launch={self.launch_s * 1e6:.2f} us, l2={self.l2_cost:.2f})")


def _clamp(value, low, high, fallback):
    if value is None:
        return fallback
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value) or value <= 0.0:
        return fallback
    return min(max(value, low), high)


class Tile:
    """One candidate configuration for one projection.

    ``kind`` is ``"dot"`` for the tensor-core tile kernel and ``"vec"`` for
    the batch-1 kernel, which is a plain FP32 FMA loop: at a single row the
    tile kernel pads x to 16 rows and pays the tensor-core layout conversion
    on every iteration for one useful row.
    """

    __slots__ = ("bm", "bn", "bk", "split", "stages", "warps", "kind", "fixup", "score")

    def __init__(self, bm, bn, bk, split=1, stages=3, warps=4, kind="dot", fixup=True):
        self.bm = int(bm)
        self.bn = int(bn)
        self.bk = int(bk)
        self.split = int(split)
        self.stages = int(stages)
        self.warps = int(warps)
        self.kind = kind
        # Whether the split-K reduction happens inside the same launch (the
        # last program to finish a column tile folds the partials) or in a
        # second kernel.  Only meaningful when ``split > 1``.
        self.fixup = bool(fixup) and int(split) > 1
        self.score = float("inf")

    @property
    def key(self):
        return (self.kind, self.bm, self.bn, self.bk, self.split,
                self.stages, self.warps, self.fixup)

    def __eq__(self, other):
        return isinstance(other, Tile) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        tag = f"{self.kind}-{self.bm}x{self.bn}x{self.bk}"
        if self.split > 1:
            tag += f"-k{self.split}" + ("f" if self.fixup else "r")
        return f"{tag}/s{self.stages}w{self.warps}"


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------

def smem_bytes(tile, glu):
    """Shared memory one block asks for, as Triton allocates it.

    The pipeliner keeps ``stages`` copies of each operand tile it loads in the
    loop.  The dot path stages x and w (and a second w for a GLU role, whose
    gate and up columns stream side by side); the vector path stages only w,
    since x is a row that lives in registers.
    """
    if tile.kind == "vec":
        per_stage = tile.bn * tile.bk * 2 * (2 if glu else 1)
    else:
        per_stage = (tile.bm * tile.bk + tile.bn * tile.bk * (2 if glu else 1)) * 2
    return per_stage * tile.stages


def blocks_per_sm(tile, dev, glu):
    """How many of these blocks fit on one SM, or 0 if one does not fit."""
    need = smem_bytes(tile, glu)
    if need > dev.smem_per_block:
        return 0
    by_smem = dev.smem_per_sm // max(need, 1)
    # 64 warps per SM on Hopper, and Triton will not co-schedule more than a
    # handful of these anyway.
    by_warps = 64 // max(tile.warps, 1)
    return max(0, min(by_smem, by_warps, 8))


def wave_efficiency(programs, resident):
    """Fraction of the machine a grid of ``programs`` blocks keeps busy.

    A grid of 160 blocks on 132 resident slots runs one full wave and one
    21%-full wave, so it delivers 61% of the device -- which is what the two
    ``N = hidden`` projections did at ``BLOCK_N = 16`` in every engine up to
    v11.  Deeper occupancy smooths this out, because the scheduler backfills
    a finished slot, so the penalty is damped rather than a cliff.
    """
    if programs <= 0 or resident <= 0:
        return 0.0
    waves = math.ceil(programs / resident)
    ideal = programs / (waves * resident)
    # Blocks do not run in lockstep: some of the tail overlaps the next wave.
    # Half the modelled loss is the compromise between the two extremes.
    return min(1.0, 0.5 * (1.0 + ideal))


# ---------------------------------------------------------------------------
# The traffic model
# ---------------------------------------------------------------------------

def traffic(tile, m, n, k, glu, dev, parts=0):
    """Modelled bytes for one projection, split into where they come from.

    Returns ``(weight_bytes, l2_bytes, effective_bytes)``.  ``n`` is the
    number of output columns the caller asks for; a GLU role streams a second
    weight block of the same shape beside the first.  ``parts`` is how many
    partial sums of squares the RMSNorm prologue reads -- every column-tile
    program reads all of them, so a producer that leaves 160 partials behind
    charges its consumer for 160, which is the second reason to prefer a wide
    tile in a producing role.
    """
    blocks = 2 if glu else 1
    tiles_n = math.ceil(n / tile.bn)
    tiles_m = math.ceil(m / tile.bm)

    # Weights: every element once per m-tile.  With m <= BLOCK_M -- the decode
    # case -- that is exactly once.
    w_bytes = tiles_m * n * k * 2 * blocks
    # x: every column-tile program streams the rows it owns over the whole of
    # K, no matter how K is split between them.
    x_bytes = tiles_n * m * k * 2
    # Partials: one FP32 copy of the output per slice, written then read.
    p_bytes = 0 if tile.split == 1 else 2 * tile.split * m * n * 4 * blocks
    # The RMSNorm hand-off, read once per program.
    s_bytes = tiles_m * tiles_n * tile.split * parts * m * 4
    # A GLU role streams two weight blocks but writes one output: the
    # epilogue multiplies them together before it stores.
    out_bytes = m * n * 2
    l2_bytes = x_bytes + s_bytes
    stream = w_bytes + p_bytes + out_bytes
    return w_bytes, l2_bytes, stream + dev.l2_cost * l2_bytes


def starvation(tile, k, glu, occupancy):
    """How much slower than its traffic a tile runs because it is not keeping
    enough bytes in flight.

    Two ways a streaming kernel goes latency-bound instead of
    bandwidth-bound, both multiplicative and both mild: too few outstanding
    bytes per SM, and a K slice with fewer iterations than the pipeline has
    stages, so the prologue is the whole kernel.  These are the terms that
    stop the model from recommending a tile that moves the fewest bytes and
    then waits for all of them.
    """
    factor = 1.0
    inflight = tile.bn * tile.bk * 2 * (2 if glu else 1) * tile.stages * occupancy
    if inflight < 64 * 1024:
        factor *= 1.0 + 0.5 * (1.0 - inflight / (64 * 1024))
    iters = (k // max(tile.split, 1)) / tile.bk
    if iters < tile.stages:
        factor *= 1.0 + 0.35 * (tile.stages - iters) / tile.stages
    return factor


def launches(tile):
    """Kernels this configuration launches: two only when K is split and the
    reduction did not fit in the same launch."""
    return 2 if (tile.split > 1 and not tile.fixup) else 1


def model_time(tile, m, n, k, glu, dev, parts=0):
    """Modelled seconds for one launch of this projection.

    Time is effective bytes over achieved bandwidth, divided by the fraction
    of the device the grid fills and multiplied by whatever starvation the
    shape implies, plus the launches it needs.  Its only job is to order a
    shortlist, not to predict a number.
    """
    occupancy = blocks_per_sm(tile, dev, glu)
    if occupancy == 0:
        return float("inf")
    tiles_n = math.ceil(n / tile.bn)
    programs = math.ceil(m / tile.bm) * tiles_n * tile.split
    resident = dev.sms * occupancy
    _, _, eff = traffic(tile, m, n, k, glu, dev, parts)
    seconds = eff / (dev.bw * wave_efficiency(programs, resident))
    return seconds * starvation(tile, k, glu, occupancy) + launches(tile) * dev.launch_s


# ---------------------------------------------------------------------------
# The candidate space
# ---------------------------------------------------------------------------

def block_m_for(m):
    """The row tile.  One m-tile if the batch fits in one, so the weights are
    read once; ``tl.dot`` needs at least 16 rows and masked rows are free."""
    bm = DOT_MIN_M
    while bm < m and bm < MAX_BLOCK_M:
        bm *= 2
    return bm


def warps_for(bn, bk, glu):
    """Warps per block, derived rather than searched.

    The model cannot separate warp counts -- they change neither traffic nor
    wave efficiency -- so sweeping them only spends shortlist slots that a
    real alternative could have used.  One warp per 2 KB of the weight tile
    keeps each thread's slice of the load wide enough to coalesce and narrow
    enough to stay in registers, which is the only thing this knob decides
    for a streaming kernel.
    """
    tile = bn * bk * (2 if glu else 1)
    if tile >= 16384:
        return 8
    if tile <= 2048:
        return 2
    return 4


def _split_options(n, k, bk, bn, dev, occupancy, allow_split, max_split):
    """K slice counts worth trying for a tile.

    A slice has to be a whole number of ``BLOCK_K`` iterations -- a masked
    tail would read weights that contribute nothing -- and there is no point
    splitting a grid that already fills the device.
    """
    if not allow_split:
        return [1]
    tiles_n = math.ceil(n / bn)
    resident = dev.sms * max(occupancy, 1)
    options = [1]
    if tiles_n >= 2 * resident:
        return options
    s = 2
    while s <= min(max_split, MAX_SPLIT):
        span = k // s
        if k % s == 0 and span % bk == 0 and span // bk >= 2 and tiles_n * s <= 4 * resident:
            options.append(s)
        s *= 2
    return options


def candidates(m, n, k, glu, dev, limit=8, incumbent=None, allow_split=True,
               allow_fixup=True, max_parts=None, max_split=MAX_SPLIT,
               partial_capacity=None, parts=0):
    """Tiles for one projection, best modelled time first.

    ``max_parts`` caps ``n / BLOCK_N`` for a role that hands its column-tile
    partial sums of squares to the next RMSNorm; ``partial_capacity`` is the
    number of FP32 elements the split-K partial buffer holds, and a tile whose
    partials would not fit is not offered.  ``incumbent``, when given, is put
    at the head of the list whatever the model thinks: the tuner starts from
    what already worked and has to be persuaded off it.
    """
    space = []
    bm = block_m_for(m)

    if m == 1:
        # The batch-1 vector kernel: no tensor cores, no row padding.
        for bn in (16, 32, 64, 128, 256):
            for bk in (32, 64, 128):
                if bn * bk > VEC_ACC_MAX or n % bn:
                    continue
                for stages in (2, 3, 4):
                    space.append(Tile(1, bn, bk, 1, stages, warps_for(bn, bk, glu),
                                      kind="vec"))

    for bn in (16, 32, 64, 128, 256):
        if n % bn:
            continue
        for bk in (32, 64, 128, 256):
            if k % bk:
                continue
            warps = warps_for(bn, bk, glu)
            for stages in (2, 3, 4, 5):
                probe = Tile(bm, bn, bk, 1, stages, warps)
                occupancy = blocks_per_sm(probe, dev, glu)
                if occupancy == 0:
                    continue
                for split in _split_options(n, k, bk, bn, dev, occupancy,
                                            allow_split, max_split):
                    for fixup in ((True, False) if (split > 1 and allow_fixup)
                                  else (False,)):
                        space.append(Tile(bm, bn, bk, split, stages, warps,
                                          fixup=fixup))

    kept = []
    seen = set()
    for tile in space:
        if tile.key in seen:
            continue
        seen.add(tile.key)
        if max_parts is not None and math.ceil(n / tile.bn) > max_parts:
            continue
        if tile.split > 1 and partial_capacity is not None:
            need = (tile.split * math.ceil(m / tile.bm) * tile.bm * n
                    * (2 if glu else 1))
            if need > partial_capacity:
                continue
        tile.score = model_time(tile, m, n, k, glu, dev, parts)
        if math.isfinite(tile.score):
            kept.append(tile)

    # Order by modelled time, then by the tie-breaks that cost nothing to
    # prefer: fewer launches, wider columns (less x traffic), shallower split.
    kept.sort(key=lambda t: (t.score, t.split > 1 and not t.fixup, -t.bn, t.split))

    ordered = []
    if incumbent is not None:
        for tile in kept:
            if tile.key == incumbent.key:
                ordered.append(tile)
                break
        else:
            incumbent.score = model_time(incumbent, m, n, k, glu, dev, parts)
            if math.isfinite(incumbent.score):
                ordered.append(incumbent)
    for tile in kept:
        if len(ordered) >= limit:
            break
        if not any(tile.key == chosen.key for chosen in ordered):
            ordered.append(tile)
    return ordered


def fit_l2_cost(narrow, wide, m, n, k, glu, dev):
    """Solve for ``bw`` and ``l2_cost`` from two measured times.

    ``narrow`` and ``wide`` are ``(tile, seconds)`` for the same projection
    run at two column widths.  They stream identical weights, so the weight
    term cancels and what is left between them is the activation re-read --
    31.5 MB against 3.9 MB for the QKV projection -- which is exactly the
    quantity ``l2_cost`` prices.  Writing each measurement as the model
    computes it and dividing one by the other gives one equation in one
    unknown:

        c = (S_n - R S_w) / (R * l_wide - l_narrow),   R = busy_n en / busy_w ew

    where ``S`` is each tile's streamed traffic -- weights, partials and the
    output, all charged at the HBM price -- and ``busy`` is the measured time
    with the launch charge removed and the starvation factor divided out.
    The bandwidth then falls out of either measurement.

    Returns ``(bw, l2_cost)``; either may be ``None`` when the pair does not
    separate the two terms well enough to be worth believing.
    """
    (tn, time_n), (tw, time_w) = narrow, wide
    on = blocks_per_sm(tn, dev, glu)
    ow = blocks_per_sm(tw, dev, glu)
    if not on or not ow or time_n <= 0 or time_w <= 0:
        return None, None
    en = wave_efficiency(math.ceil(m / tn.bm) * math.ceil(n / tn.bn) * tn.split,
                         dev.sms * on)
    ew = wave_efficiency(math.ceil(m / tw.bm) * math.ceil(n / tw.bn) * tw.split,
                         dev.sms * ow)
    if not en or not ew:
        return None, None
    busy_n = (time_n - launches(tn) * dev.launch_s) / starvation(tn, k, glu, on)
    busy_w = (time_w - launches(tw) * dev.launch_s) / starvation(tw, k, glu, ow)
    if busy_n <= 0 or busy_w <= 0:
        return None, None

    # Traffic that does not depend on the unknown.  It is not the same for
    # both tiles once one of them splits: the partials are charged at the
    # stream price, so they belong on this side of the equation.
    def split_terms(tile):
        _, l2_bytes, eff = traffic(tile, m, n, k, glu, dev)
        return l2_bytes, eff - dev.l2_cost * l2_bytes

    ln, sn = split_terms(tn)
    lw, sw = split_terms(tw)
    ratio = (busy_n * en) / (busy_w * ew)

    l2_cost = None
    denom = ratio * lw - ln
    if abs(denom) > 0.05 * max(ln, 1.0):
        candidate = (sn - ratio * sw) / denom
        if math.isfinite(candidate) and 0.0 < candidate <= 1.0:
            l2_cost = candidate
    price = l2_cost if l2_cost is not None else dev.l2_cost
    bw = (sw + price * lw) / (busy_w * ew)
    return bw, l2_cost


def describe(tile, m, n, k, glu, dev, parts=0):
    """One line for the run log: what the model thinks this tile costs."""
    w, l2, eff = traffic(tile, m, n, k, glu, dev, parts)
    occupancy = blocks_per_sm(tile, dev, glu)
    programs = math.ceil(m / tile.bm) * math.ceil(n / tile.bn) * tile.split
    return (f"{tile} programs={programs} occ={occupancy} "
            f"w={w / 1e6:.1f}MB l2={l2 / 1e6:.1f}MB eff={eff / 1e6:.1f}MB "
            f"model={model_time(tile, m, n, k, glu, dev, parts) * 1e6:.1f}us")
