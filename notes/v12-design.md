# v12: the decode step re-reads its activations 6.25 GB per step

Branch `engine-v12`, on top of v11. One diagnosis, four changes, no new
kernels.

## What v11's run actually told us

v11 (`91b25bef`, 898.54) was built around wave quantisation: the O and down
projections tile 2560 columns into 160 programs on 132 SMs, 61% occupancy.
It searched a far wider tile space than v8 and came back with B16 TPOT of
4.803 ms against v8's 4.808. Identical. The premise was either wrong or
unreachable, and the run could not say which because official runs return
no engine stdout.

Two things were wrong, and they compound.

### 1. The cost model had no term for activation re-reads

A projection launches `N / BLOCK_N` programs, and **each one streams the
whole `[BLOCK_M, K]` input tile**. At `BLOCK_M = BLOCK_N = 16` -- the tile
v1 through v11 all ran -- the x tile and the weight tile per K-iteration are
both 8192 bytes. Half of every load is a re-read of the same 80 KB.

| role | N | K | weights | x re-reads | ratio |
|---|---:|---:|---:|---:|---:|
| qkv | 6144 | 2560 | 31.5 MB | 31.5 MB | 1.00 |
| o | 2560 | 4096 | 21.0 MB | 21.0 MB | 1.00 |
| gate/up | 9728 | 2560 | 99.6 MB | 49.8 MB | 0.50 |
| down | 2560 | 9728 | 49.8 MB | 49.8 MB | 1.00 |
| **per layer** | | | **201.9 MB** | **152.0 MB** | |
| lm | 151936 | 2560 | 777.9 MB | 777.9 MB | 1.00 |

Per step: **8.05 GB of weights and 6.25 GB of activation re-reads.** x is
80 KB, so it lives in L2 and never appears as HBM traffic -- it appears as a
step that issues 14.30 GB of loads to deliver 8.05 GB.

That reframes the headline number. At the measured 4.803 ms the step is not
running at 1.97 TB/s of a 3.35 TB/s device. It is running **14.30 GB /
4.803 ms = 2.98 TB/s of aggregate load throughput**, which is close to what
the memory system can actually sustain. The engine was never 40% off peak.
It was doing 78% more work than necessary, at close to peak.

This also explains the two previous results that never fit. v4 added split-K
to O and down and gained 1.3%; v9 extended it to the other three and gained
nothing. Split-K alone fixes wave quantisation without touching the x term,
and qkv, gate/up and lm already had full grids -- so there was nothing for it
to do. The win needs a **wide tile and split-K together**: the wide tile cuts
the re-reads and empties the device, the split fills it back up, and each
split program reads only `K / SPLIT` of the row so the x term does not come
back.

### 2. The tuner could not have found it anyway

`_race` required a candidate to beat the incumbent by 3%, applied
independently to each of five roles. That 3% is the platform's *between-run*
noise. A tile race is an in-process A/B on the same device in the same
second, over 36 layers of real weights, minimum-of-rounds -- its noise is far
below 1%. Using the between-run figure there rejects a 2% win on every role
while each rejection looks individually defensible, and forfeits the product.

The shortlist made it moot: `BLOCK_N` never exceeded 32 in any role, and for
qkv and gate/up all nine candidates were `BLOCK_N = 16`. Ties in wave
efficiency were broken toward the *narrow* tile, which is the direction that
maximises re-reads.

## Changes

1. **`_score` ranks modelled traffic, not wave efficiency.**
   `(weights + 0.6 * (x re-reads + split partials)) / wave_efficiency`, plus
   a launch charge for the split-K reduce, a penalty for too few bytes in
   flight, and a penalty when a deep split leaves fewer K-iterations than
   pipeline stages. L2 bytes are charged at 0.6x HBM bytes: cheaper per byte,
   not free, because they contend for the same issue slots and L2 ports.
2. **Split-K is gated on the candidate's own grid** (`n // cfg.bn <
   2 * SMs`), not on the 16-wide grid. This is the change that makes wide
   tiles viable. `BLOCK_K = 64` joins the space because `K = 9728` is
   `512 * 19`, so at `bk = 128` the down projection cannot split past 4.
3. **`TILE_MARGIN = 0.995` for the per-role race**, with rounds raised 2 -> 4.
   `MARGIN = 0.97` still guards the big binary decisions (fused vs reference,
   graph vs eager) where a bad call costs the whole run.
4. **Tiles are cached per batch, not per shape.** They depend on
   `(batch, role)` alone, so six hidden workloads no longer re-spend the
   budget; that pays for raising it 60 s -> 80 s and the shortlist 9 -> 10.

Two batch-1 fixes fall out of the same review. The incumbent tile was not
offered at all at `m == 1`, so v11's new vector kernel ran with nothing to
beat -- and batch-1 TPOT drifted 3.883 -> 3.918 -> 3.964 -> 3.990 ms across
v8, v10, main and v11, the one signal in the run data that is not noise. It
is now offered and has to be beaten like anywhere else. Separately, the model
charged the tile kernel for all `BLOCK_M` rows when only `m` of them issue
loads; masked rows are free, and at batch 1 that was a 16x overcharge against
the tile kernel.

## What the model predicts

| batch | modelled speedup | what it picks |
|---|---:|---|
| 1 | 1.16x | vector kernel for qkv/o/gate-up, `k8-32x64` down, `128x64` lm |
| 4 | 1.18x | `16x64` qkv, `k16-64x64` o, `k8-32x64` down, `128x64` lm |
| 16 | 1.31x | `k8-128x64` qkv, `k16-64x64` o, `k8-32x64` down, `128x64` lm |
| 32 | 1.47x | as 16, wider gate/up |
| 64 | 2.12x | wide everywhere |

At batch 16 that is TPOT 4.80 -> 3.67 ms and a score near 1180. Treat the
number as a direction, not a forecast: it is a model with two fitted
constants (`L2_COST`, `LAUNCH_BYTES`), and its only job is to order the
shortlist so that good candidates get compiled inside the budget. The tuner
still measures every one and still starts from the tile that already worked.

The corroboration worth noting is that #1 scores 1280, which on this
arithmetic is a step at 2.79 TB/s of *useful* traffic -- i.e. roughly what
this engine already achieves in aggregate, once it stops spending 44% of it
re-reading the same 80 KB. No algorithmic change is needed to explain the
leaders.

## Not done

- **Split-K for gate/up.** Two accumulators need two partial buffers and a
  changed reduce. Modelled at 1.10x for that role alone, and it is the one
  role whose x term is already halved because gate and up share an x tile.
- **A persistent / grid-strided kernel**, which would make wave quantisation
  irrelevant and let `BLOCK_N` be chosen on traffic alone. This is the next
  structural move if v12's measurement confirms the diagnosis.
- **Profiling.** None of this was measured on a GPU. `ncu` on one rented
  H100 hour would confirm or kill the diagnosis directly, instead of through
  three public numbers per 10-minute run.

## Verification

`python tests/test_proj_plan.py` (9 tests) and
`python tests/test_fused_wiring.py` (4 tests) pass. New coverage: that a wide
tile is offered for every role that can use one, that anything which splits
still fills the device and still fits the partial buffer, and that split-K is
only offered where that tile's own grid is short.

The partial buffer is now sized from the widest role allowed to split (QKV's
6144 columns, not hidden's 2560), and `project()` refuses a split whose
partials would not fit rather than writing past it.
