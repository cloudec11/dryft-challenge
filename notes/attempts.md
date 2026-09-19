# Attempts log

One entry per platform run. Score = geometric mean tok/s over the 6 hidden
workloads. Fill in the numbers from the run page.

| # | Commit | Change | Score (tok/s) | Pass? | Notes |
|---|--------|--------|---------------|-------|-------|
| 0 | 186912a | Baseline (Transformers, unchanged) | _hidden score?_ | yes | samples below |
| 1 | 5aed25f | v1: hand-rolled forward, static KV cache, CUDA-graph decode, fused Triton kernels | 855.3 | yes | official run c0c37a87, leaderboard #25; samples below |
| 2 | daedece | v2: fused skinny-GEMM decode step (6 launches/layer), GQA flash prefill, row qkv_post, 3-step lookahead | 882.2 | yes | official run 4e9a638e, +3.1% over v1 |
| 3 | 94660b0 | v3: prefill CUDA graph (raced), single-row GEMV for B1, lookahead 1 before first token | 882.7 | yes | official run 3bdf08de; B1 +13% but score flat -> hidden shapes are not batch 1 |
| 4 | beb0732 | v4: split-K GEMV for the N=2560 projections, fused step tried up to batch 128 | 894.5 | yes | run a93ebf2b, +1.3%; TPOT down at every public shape |
| 5 | 912d2ba | v5: decode attention tile/split tuned at warmup | 882.2 | yes | run b4494a7a, **-1.4% regression**: the benchmark was L2-resident |
| 6+7 | 6505b80 | v6 single-split attention + v7 split-8/argmax + cold-KV tuner fix | 888.2 | yes | run b54796c8; B16 best yet (715.8 ms), B1/B4 -2% |
| 8 | cd5ef7c | v8: noise-robust tuner decisions (3% margin, min-of-rounds) | **902.9** | yes | run 84daa14e, **best so far, #22**; all three public shapes improved together |

## v1 design (branch `fast-engine`)

- Weights packed at load: QKV -> one [6144, 2560] GEMM, gate/up -> one [19456, 2560].
- Prefill: eager, same SDPA/FlashAttention call as the baseline on GQA-expanded K/V.
- Decode: one CUDA graph per shape, captured in the warmup call. Token and
  position live on the GPU; the graph advances the position itself, so step
  t+1 is queued before step t's tokens are yielded.
- Triton: add+RMSNorm, head-norm+RoPE+cache write, SwiGLU, split-K GQA decode
  attention. Each is checked against a PyTorch twin at load time and replaced
  by the twin if it disagrees (see `[engine] kernels:` in the run log).
- Verified locally on CPU only (tests/test_engine_cpu.py): identical tokens to
  the Transformers loop in FP32; BF16 replay margin <= 0.03.

What to read in the run log: `[engine] kernels: ...` (which kernels are live),
`graph=yes`, and per-shape setup time.

## Baseline sample cases (run 0)

| Workload | TPS | Batch time | TTFT | TPOT | Memory |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 39.5 | 811 ms | 29.8 ms | 25.2 ms | 9.57 GiB |
| B4 2048->32 | 115.6 | 1107 ms | 202.6 ms | 29.2 ms | 11.44 GiB |
| B16 512->128 | 582.8 | 3514 ms | 191.7 ms | 26.2 ms | 11.44 GiB |

TPOT is ~25-29 ms at every batch size: decode is pure launch overhead
(the weight read alone is ~3 ms). Prefill is ~30 ms (B1) to ~200 ms (B4x2048).

## v1 sample cases (run 1, official c0c37a87)

| Workload | TPS | Batch time | TTFT | TPOT | Memory | vs baseline TPS |
|---|---:|---:|---:|---:|---:|---:|
| B1 512->32 | 217.2 | 147.3 ms | 14.35 ms | 4.29 ms | 10.23 GiB | 5.5x |
| B4 2048->32 | 449.6 | 284.7 ms | 134.17 ms | 4.87 ms | 12.65 GiB | 3.9x |
| B16 512->128 | 2733.0 | 749.4 ms | 122.61 ms | 4.93 ms | 12.91 GiB | 4.7x |

Where the time goes (batch time = TTFT + (N-1) * TPOT):

- B1: decode 133 ms of 147 (90%). Prefill 14 ms for 512 tokens is ~2x its
  compute floor: launch overhead and small-M GEMMs, not FLOPs.
- B4x2048: prefill 134 ms of 285 (47%). 8192 tokens at ~490 TFLOP/s, i.e.
  already cuBLAS-bound; only ~10-15% left there.
- B16: decode 626 ms of 749 (84%).

Decode floor: weights ~8.05 GB (7.27 GB layers + 0.78 GB tied lm_head) plus
the KV read (~1.4 GB/step at B16x640, ~1.2 GB at B4x2080). On H100 SXM
(3.35 TB/s) that is ~2.4 ms at B1 and ~2.8 ms at B16, so v1 sits at ~55-60%
of bandwidth. On H100 PCIe (2.0 TB/s) the B1 floor is ~4.0 ms and v1 is
already at ~94%. Which one we are on decides the next move; log
torch.cuda.get_device_name() and a replay timing in the next public run.

Latency gates have large headroom: TTFT is 0.5-0.7x native, TPOT is ~0.17x.

## v2 design (branch `fast-engine`)

Target: decode step time (84-90% of B1/B16) and prefill glue.

- `kernels/fused.py`, a second decode step, 6 launches per layer instead of 10:
  - QKV GEMV with the input RMSNorm in its prologue.
  - Attention split kernel with q/k head norm + RoPE + KV-cache write folded
    in (the new k/v replace slot `pos` in registers; the split owning `pos`
    stores them). The reduce kernel is unchanged.
  - O GEMV whose epilogue does the residual add in place and writes per-program
    partial sums of h^2; gate/up GEMV reads those to normalise in its prologue
    and applies SwiGLU in its epilogue; down GEMV is like O.
  - Embedding + first sum of squares in one kernel; LM head is the same GEMV
    with the final norm in its prologue, then torch.argmax.
  - BF16 rounding points identical to the reference; only reduction order
    differs (tile sums, FP32 partial sums of squares).
- Warmup picks the fused step only if (a) its logits match the v1 step within
  1.5 max / 0.1 mean on a synthetic prompt of the real shape over 3 steps and
  (b) its captured graph replays faster than v1's. So it can't be slower
  than v1's decode.
- GEMV tiles tuned per projection at warmup (6 candidates, 36-layer graph so
  weights don't sit in L2, 90 s budget).
- Prefill: SDPA with `enable_gqa=True` forced onto FlashAttention (no
  repeat_kv copies, no q transpose copy); load-time check vs the expanded call,
  falls back if unsupported. qkv_post now runs one program per 16 heads of a
  row (3 per token instead of 48 one-warp programs).
- Generate keeps 3 graph replays queued ahead of the caller (was 1).

What to read in the run log: `device ...`, `kernels: ... prefill_attn=...`,
per-role `best ... GB/s` lines, `fused step vs v1: max|dlogit|=...`,
`decode step: v1 X ms, fused Y ms -> ...`, and `step=fused|v1`.

Expected (if fused wins at ~3.4 ms/step and prefill gets ~8%): B1 ~118 ms,
B4 ~245 ms, B16 ~610 ms, i.e. ~+20% score.

## v2 sample cases (run 2, official 4e9a638e) - score 882.15

| Workload | TPS | Batch time | TTFT | TPOT | Memory | vs v1 total |
|---|---:|---:|---:|---:|---:|---:|
| B1 512->32 | 212.8 | 150.4 ms | 19.21 ms | 4.25 ms | 10.23 GiB | +2.1% slower |
| B4 2048->32 | 475.1 | 269.4 ms | 118.35 ms | 4.86 ms | 12.57 GiB | -5.4% |
| B16 512->128 | 2813.4 | 727.9 ms | 105.65 ms | 4.89 ms | 12.83 GiB | -2.9% |

Sample spread 0.5-0.6% (gate is 25%). Native ratios: TTFT 0.44-0.58x,
TPOT 0.12x. Device confirmed: **NVIDIA H100 80GB HBM3** (SXM, ~3.35 TB/s),
driver 580.95.05, gVisor sandbox, harness 0.2.0.

What we learned:

1. Prefill work paid off: TTFT -12% (B4), -14% (B16).
2. Decode barely moved (4.25 vs 4.29 ms), so the fused step either lost the
   warmup race or failed its logit check. **Official runs hide engine stdout
   and the API only accepts mode=official** (CreateRunBody pins it), so the
   `[engine]` lines are never visible: iterate on metrics alone.
3. B1 TTFT regressed 14.4 -> 19.2 ms. The only B1-specific change is the
   3-step lookahead, so a graph launch appears to cost the CPU ~2.4 ms here
   (~370 nodes, gVisor). B4/B16 hid those launches behind a long prefill.
   Corollary: a short eager prefill (~430 launches) is probably launch-bound
   too, which is why v3 graphs it.

## v3 changes

- Queue one decode step before the first token, two afterwards (LOOKAHEAD=2).
- CUDA-graph the prefill when batch * prompt <= 4096 tokens, over a static
  prompt buffer; longer prefills stay eager (GPU-bound, and their activations
  would sit in a permanent graph pool).
- Add `_gemv_vec_kernel`: at batch 1 the tile kernel pads a single row to 16
  for tl.dot; the new one does plain FP32 FMA into a [BLOCK_N, BLOCK_K]
  accumulator. Tried first at B1, tile kernel still tried as a fallback.
- Sum-of-squares buffers grow to 1024 partials (BLOCK_N can now be 4) and the
  norm-in consumers are tuned after the producers so PARTS matches.

## v3 sample cases (run 3, official 3bdf08de) - score 882.68

| Workload | TPS | Batch time | TTFT | TPOT | Memory |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 241.2 | 132.6 ms | 10.46 ms | 3.945 ms | 10.34 GiB |
| B4 2048->32 | 469.9 | 272.4 ms | 118.45 ms | 4.981 ms | 12.58 GiB |
| B16 512->128 | 2756.8 | 742.9 ms | 106.98 ms | 5.013 ms | 12.84 GiB |

- B1: prefill graph won (TTFT 19.2 -> 10.5, below v1's 14.4) and TPOT fell
  under v2's 4.25, so the single-row GEMV won the race: the fused step is
  live at batch 1. Total -12%, TPS +13%.
- B4/B16: prefill graph lost its race (memory unchanged, TTFT unchanged), and
  TPOT drifted +2%. Clocks are not locked, so treat ~2% between runs as noise.

**The score moved 0.06% while B1 moved 13%, so no hidden workload is batch 1.**
Hidden geomean TPS is 883 against 678 for the three public shapes, so the
hidden set is faster per token: larger batches and/or longer outputs, i.e.
decode-dominated. Optimize decode at batch >= 4 and prefill, not batch 1.

Decode budget at B16: weights 8.05 GB + KV ~1.4 GB per step = ~2.8 ms at
3.35 TB/s, measured 5.01 ms. Suspected waste: the O and down projections
(N=2560) tile into 160 programs over 132 SMs, so the second wave is ~21%
full, and every program re-reads all of x from L2 (52 MB per layer at M=64).
Split-K fixes both (more tiles, larger BLOCK_N).

## v4 changes

- `_gemv_splitk_kernel` + `_gemv_splitk_reduce_kernel` for the O and down
  projections (N = hidden): each program takes one K slice, writes FP32
  partials, and the reduce kernel sums them (FP32, rounded to BF16 once, like
  the unsplit GEMM) and runs the residual-add + sum-of-squares epilogue.
  Split-K tiles are tried first for those two roles: split 2/4 with BLOCK_N
  32-128 gives 40-320 tiles per slice instead of 160 unsplit, and the larger
  BLOCK_N also cuts the repeated L2 reads of x (52 MB -> 13 MB per layer at
  M=64, BLOCK_N 16 -> 64).
- Fused step now tried up to batch 128 (tiles that overflow shared memory
  just fail to compile during tuning and are skipped).
- Tuning budget is sliced per role so the producers cannot starve the rest.
- LOOKAHEAD back to 3 after the first token (v3 used 2 and B4/B16 TPOT drifted
  +2%; the first-token ramp already protects TTFT).

## v5 changes

- `ops.ATTN_CANDIDATES`: BLOCK_N 64/128/256 x target programs 132/264/528 x
  warps, raced at warmup with pos = capacity - 1 (the worst case) and checked
  against the FP32 reference before use. The old values were fixed constants
  (BLOCK_N 64, 264 programs, 4 warps).
- Rationale: at batch 32 over a 2048-token context the KV read is ~9.8 GB per
  step against 8.05 GB of weights, so at the batch sizes the hidden set seems
  to use, attention -- not the projections -- can be the larger half of the
  step. A fixed tiling is a guess; this measures it.

## v4 sample cases (run 4, official a93ebf2b) - score 894.48 (+1.34%)

| Workload | TPS | Batch time | TTFT | TPOT | Memory |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 244.3 | 131.0 ms | 11.13 ms | 3.863 ms | 10.34 GiB |
| B4 2048->32 | 472.3 | 271.0 ms | 122.23 ms | 4.806 ms | 12.93 GiB |
| B16 512->128 | 2810.5 | 728.7 ms | 108.76 ms | 4.883 ms | 13.19 GiB |

Split-K is a real win: TPOT -2.1% (B1), -3.5% (B4), -2.6% (B16), and the
hidden score moved with it (+1.34%), so the hidden shapes are decode-bound
as suspected. Memory +0.35 GiB from the FP32 partial buffer. TTFT drifted
+3% with no prefill change in this version: between-run noise is ~3%.

Where decode still stands at B16: 4.883 ms/step. Weights 8.05 GB + KV 1.4 GB
at 3.35 TB/s = 2.8 ms floor, so ~1.7x off, i.e. the projections stream at
roughly 55% of peak bandwidth. cuBLAS (v1) was no better, which points at a
structural cause rather than a bad kernel: ~220 launches per step, each
kernel short enough (9-30 us) that wave fill and drain are a large fraction
of it. That makes kernel count, not arithmetic, the thing to attack next.

## v6 changes

- Both decode attention kernels take an `ONE_SPLIT` flag: with one split the
  program already holds the whole sequence's running softmax, so it divides
  by l_i and stores the output itself. The reduce kernel is then not launched
  at all: 36 fewer launches per step. Identical arithmetic (with one split the
  reduce's weights are exactly 1).
- Two candidates with target_programs=1 added, so the tuner can pick a single
  split when batch * kv_heads already fills the device.
- Motivation from run 4: decode streams at ~55% of peak bandwidth and cuBLAS
  was no better, so the limit looks structural - ~220 short launches per step,
  each paying wave fill/drain. Cutting launches is the lever.

## v7 changes

- Split-K candidates up to 8 slices (MAX_SPLIT 8): for the O projection
  (K=4096) that is 8 x 512 columns per slice, 320 programs at BLOCK_N 64.
  The tuner skips any slice count that does not divide K by BLOCK_K.
- `torch.argmax(..., out=ids)` instead of argmax-then-copy: one less launch
  per step, probed once at load so an unsupported out= cannot break capture.

## v5 sample cases (run 5, official b4494a7a) - score 882.24 (-1.37%)

| Workload | TPS | Batch time | TTFT | TPOT | Memory |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 238.0 | 134.5 ms | 10.61 ms | 3.996 ms | 10.34 GiB |
| B4 2048->32 | 471.6 | 271.4 ms | 118.92 ms | 4.921 ms | 12.93 GiB |
| B16 512->128 | 2761.8 | 741.5 ms | 107.22 ms | 4.987 ms | 13.19 GiB |

A regression, and the cause is the benchmark, not the idea: `_tune_attention`
replayed one layer's cache back to back, and one layer's KV is 42 MB at
B16 x 640 slots, which fits in the 50 MB L2. It measured an L2-resident read
and chose a tile for a regime the real step never sees (in the step, every
layer's KV is cold). TPOT rose 2-3% at every shape, consistent with a tile
that is wrong for cold reads.

Lesson, the same one the GEMV tuner already encodes: **a warmup benchmark must
touch as much distinct memory as the real step does.** Fixed by timing one
call per layer over that layer's own cache.

## Run 6 (official b54796c8, v6+v7+tuner fix) - score 888.23

| Workload | TPS | Batch time | TTFT | TPOT | Memory |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 240.1 | 133.3 ms | 10.83 ms | 3.952 ms | 10.34 GiB |
| B4 2048->32 | 474.2 | 269.9 ms | 117.45 ms | 4.906 ms | 12.93 GiB |
| B16 512->128 | 2861.0 | 715.8 ms | 106.32 ms | 4.799 ms | 13.17 GiB |

(The run also carried a platform-side `harness_error` note on an attempt;
the measurement itself completed and scored.)

B16 is the best result so far (-1.8% batch time vs v4), consistent with the
single-split attention dropping 36 launches per step. B1 and B4 each gave
back ~2%.

### Score noise

Hidden scores so far: 882.15, 882.68, 894.48, 882.24, 888.23. Clocks are not
locked and each run measures 5 samples per shape, so **run-to-run noise is
~1-2% and any single-run delta below ~2% is not evidence.** v4's split-K win
was real because TPOT improved on all three public shapes at once; most other
deltas are not distinguishable from noise.

Consequence for the design: a tuner that takes the bare minimum of noisy
measurements will sometimes pick a worse configuration, which is the likely
cause of B1 losing its single-row GEMV to a split-K candidate in run 6.

## v8 changes

- Every warmup race now needs a 3% win to move off the earlier (safer)
  candidate: GEMV tiles, attention tiles, fused-vs-v1, prefill graph-vs-eager.
  Candidate lists are ordered so earlier means "what already worked".
- Timings are the min over 2 rounds of 2 replays rather than the mean of 3,
  so one disturbed round cannot decide a race.

## Run 7 (official 84daa14e, v8) - score 902.88, leaderboard #22

| Workload | TPS | Batch time | TTFT | TPOT | Spread |
|---|---:|---:|---:|---:|---:|
| B1 512->32 | 244.4 | 130.9 ms | 10.55 ms | 3.883 ms | 0.7% |
| B4 2048->32 | 482.2 | 265.4 ms | 116.74 ms | 4.807 ms | 0.9% |
| B16 512->128 | 2858.8 | 716.4 ms | 105.09 ms | 4.808 ms | 0.1% |

All three public shapes improved at once (B1 -1.8%, B4 -1.7%, B16 flat vs
run 6), which is the signature of a real change rather than noise. The
margin recovered what run 6 gave back at B1/B4 while keeping the
single-split attention win at B16.

Cumulative: 855.3 (v1) -> 902.9 (v8), +5.6%, and 22.9x the Transformers
baseline's public-shape throughput at B1 (39.5 -> 244.4 tok/s... 6.2x;
B16 582.8 -> 2858.8, 4.9x).

## Where the remaining time is (v8, B16)

Per step 4.808 ms against a ~2.8 ms floor (8.05 GB weights + 1.4 GB KV at
3.35 TB/s), so ~58% of peak bandwidth. ~185 launches per step, each 9-30 us,
so wave fill/drain and launch gaps are an estimated 0.8-0.9 ms of the 2.0 ms
gap. cuBLAS was no better, so this is structural to one-kernel-per-stage
decoding, and the per-stage decomposition is already minimal (5 launches per
layer: QKV, attention, O, gate/up, down).

Ideas left, both structural and both risky:

1. **Speculative decoding** (prompt-lookup n-gram drafts, exact verification).
   Legal and correct by construction. Verifying K+1 tokens costs almost the
   same as 1 at these batch sizes because the step is weight-bound, so any
   acceptance is a win. Risk: acceptance varies per prompt, so sample times
   vary, and >25% spread fails the run (best score is kept, so the cost is a
   wasted run). Needs per-sequence ragged positions in the cache, attention,
   RoPE and cache writes: a moderate rewrite with no local GPU to debug on.
2. **Decode megakernel** (one launch per layer or per step). Removes the
   launch/ramp overhead, but Triton has no grid barrier; spin-waiting on
   blocks that may not be co-resident can deadlock, and a hang burns the
   300 s sample limit.

## v9 changes

Target: decode streams at 58% of peak. Besides the HBM weight stream, each
step re-reads its activations out of L2 once per column tile: at BLOCK_N=16
gate/up re-reads x 608 times (48 MB), down 160 times (50 MB), ~5.4 GB of L2
traffic per step. Since every HBM read also passes through L2, that is close
to a second bottleneck. Wider tiles fix it but cost programs -- unless K is
split.

- Split-K generalized from the two residual projections to every role but the
  LM head: the split kernel now carries the RMSNorm prologue and writes gate
  and up slices for the GLU role; the reduce kernel carries all three
  epilogues (GLU, residual+sum-of-squares, plain).
- Candidate order per role keeps v8's margin discipline: split-K leads for O
  and down (where run 4 proved it), and has to win by 3% for QKV and gate/up.
- Partial buffer is now [2 * MAX_SPLIT, BLOCK_M, intermediate] FP32 (10 MB at
  batch 16), the widest any role needs.
