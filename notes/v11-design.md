# v11: rewritten from scratch around the decode step's real costs

Branch `engine-v11`. The v10 engine and its kernels are kept verbatim under
`notes/v10/` so the two can be compared or reverted to.

## Why rewrite at all

v8 scored 902.9 (#22) and v10 scored 900.9, which is the same run inside
this platform's 1-2% noise. The gap to #1 (1144) is +27%, and no tile
tweak reaches that, so it is worth writing down where the time actually is
before changing anything.

At B16, 512 -> 128, TPOT was 4.808 ms. What the step must move:

| | bytes | at 3.35 TB/s |
|---|---:|---:|
| weights (7.27 GB layers + 0.78 GB tied LM head) | 8.05 GB | 2.40 ms |
| KV read at batch 16 over 640 slots | 1.40 GB | 0.42 ms |
| ~185 launches at a measured ~2.8 us | | 0.52 ms |
| **accounted** | 9.45 GB | **3.34 ms** |
| **measured** | | **4.81 ms** |

So the step streams at **2.0 TB/s effective, 60% of the device**, and the
same number falls out at batch 1 (8.13 GB / 3.883 ms = 2.09 TB/s). That
consistency is the useful part: the shortfall is not about batch size, it
is about how the projections stream.

The ~2.8 us per launch is not a guess. v6 removed 36 launches per step by
letting a single-split attention skip its reduce kernel, and B16 batch time
fell 13 ms over 127 steps: 0.1 ms per step / 36 launches.

### Where the missing 1.5 ms goes

Wave quantisation, mostly. A projection launches `N/BLOCK_N` programs over
132 SMs, and with `BLOCK_N=16`:

| role | N | programs | rounds | efficiency |
|---|---:|---:|---:|---:|
| qkv | 6144 | 384 | 2.91 -> 3 | 97% |
| **o** | **2560** | **160** | **1.21 -> 2** | **61%** |
| gate/up | 19456 | 1216 | 9.21 -> 10 | 92% |
| **down** | **2560** | **160** | **1.21 -> 2** | **61%** |
| lm | 151936 | 9496 | 71.9 -> 72 | 99% |

The two `N = hidden` projections are 33% of the layer's weight bytes and
run at 61% occupancy. That is most of the missing bandwidth, and it is
exactly what v4's split-K happened to fix for those two roles (+1.3%), and
exactly why v9's extension of split-K to the other three did nothing: they
were already balanced.

## What the rewrite changes

Same proven skeleton -- packed weights, static KV cache, CUDA-graphed decode
captured at warmup, flash-GQA prefill raced against a prefill graph, five
launches per layer, noise-robust races with a 3% margin. What is new:

1. **One projection kernel instead of four.** `kernels/proj.py` has a single
   kernel specialised on `NORM` / `GLU` / `RES` / `SPLIT`, with the split-K
   reduce sharing the same epilogue function, so the fused and split paths
   cannot drift apart.

2. **Tiles derived from the device, not from a list.** `proj.candidates`
   enumerates `BLOCK_N x BLOCK_K x split x stages x warps`, drops anything
   that does not divide the matrix or fit in shared memory, and sorts by
   wave efficiency (coarse buckets) then bytes in flight per SM. Split-K is
   offered only where `N/16 < 2 * SMs`, i.e. only for O and down. The
   incumbent (16x256, the tile v1-v10 ran) still leads the list and still
   has to be beaten by 3%.

   Two knobs the old fixed list never reached: `BLOCK_K` up to 1024 and
   `num_stages` 4, which is the lever on bytes in flight per SM -- the other
   candidate explanation for 60% of peak.

3. **Producers prefer wide tiles.** A `RES` epilogue writes one sum-of-
   squares partial per program, and every consumer's RMSNorm prologue reads
   all of them. At `BLOCK_N=16` that is 160 partials, ~10 KB re-read by each
   of ~67k consumer programs per step, about 1% of the step. Ties in wave
   efficiency now break towards the wider tile.

4. **Instrumentation that localises the rest.** Every shape logs
   `step X ms = Y GB/s over Z GiB`, and every role logs its achieved GB/s,
   program count, wave efficiency and bytes in flight. One public run now
   says how much of the 40% is left and which role owns it.

5. **Speculation dropped.** Sample prompts are freshly random token ids, so
   a prompt-lookup n-gram draft has almost nothing to match; v10 measured
   flat against v8, which is consistent with its own gate turning it off.
   It was ~300 lines of the riskiest code in the engine (ragged per-sequence
   positions through RoPE, the cache write and attention) guarding a win
   that never arrived.

## What was considered and rejected

- **Folding the split-K reduce into the consumer's prologue.** It would
  remove 72 launches per step (~0.2 ms), but each consumer program would
  then re-read the FP32 partials: 4 slices x 16 rows x 2560 x 4 B is 655 KB
  per program, ~196 MB of L2 traffic per call against the 0.3 MB the reduce
  kernel reads once. Roughly 20 us to save 2.8 us.
- **A whole-layer megakernel** (1 launch per layer instead of 5, ~0.4 ms).
  Needs a grid-wide barrier, which Triton does not have; doing it by hand
  requires a grid no larger than the number of co-resident blocks, and
  getting that wrong is a hang, which costs the 300 s sample limit and the
  whole run. Not worth it without a GPU to debug on.
- **Fusing argmax into the LM head.** Saves the 4.9 MB logits round trip:
  about 3 us of 4800. Not worth the tie-breaking risk.
- **Prefill.** B4x2048 prefill is 117 ms against a ~73 ms compute floor, so
  there is maybe 3% of total there, and it is cuBLAS-bound. Left alone.

## Expected size of the win

Honest estimate: the wave-efficiency work is mostly what v4 already found,
so the new part is the wider tuning space (BLOCK_K, stages, warps) plus the
producer-width tie-break. That is a few percent, not +27%. The rewrite's
larger value is that the next run's log says where the remaining 40% is,
instead of leaving it to be inferred from three public numbers.

If the achieved GB/s comes back near 2.0 TB/s with good wave efficiency
everywhere, the limit is memory-level parallelism inside the kernel and the
next move is the megakernel or a persistent grid. If it comes back at 2.6+
TB/s, launches are the remaining cost and the megakernel is the only lever
left.

## Verification without a GPU

- `python tests/test_proj_plan.py` -- every shortlisted tile divides the
  matrix, fits shared memory at every batch from 1 to 128, and cannot
  overflow the sum-of-squares hand-off buffer; split-K is offered only where
  intended; the incumbent leads. (Found one real bug: the incumbent was
  prepended without its own shared-memory check and does not fit at batch
  128.)
- `python tests/test_fused_wiring.py` -- runs the real `_decode_fused` over
  a tiny random model with the Triton calls swapped for PyTorch equivalents
  that reproduce the partial layout, and requires its logits to match
  `_decode_ref`. Also checks the generate contract and that two consecutive
  calls share no state.
- `python tests/test_engine_cpu.py` -- needs `transformers==4.51.3`; checks
  tokens against the starter's Transformers loop and runs a judge-style
  teacher-forced replay.

Nothing here exercises a Triton kernel. The BF16 rounding points are the
reference's by construction, and each kernel is checked against its twin at
load time on the GPU (`[engine] kernels: ...` in a public run's log).

## What to read in the first public run

```
[engine] device ..., 132 SMs
[engine] kernels: add_norm=triton, qkv_post=triton, ... decode_attn=triton
[engine]   o n=2560 k=4096: k4-32x256s3w4 at ...us | ...
[engine]     o: NNNN GB/s, 640 programs, 0.97 wave efficiency, NN KiB in flight/SM
[engine]   attn: bn64x1w4s3 at ...us      <- 1 split means no reduce launch
[engine] fused step vs reference: max|dlogit|=... -> ok
[engine] decode step: reference X ms, fused Y ms -> fused
[engine] shape B=16 S=512 N=128: ... step 4.xxx ms = NNNN GB/s over 8.80 GiB
```

The last line is the one that matters. 2000 GB/s means nothing changed;
2600+ means the wave work paid and launches are what is left.
