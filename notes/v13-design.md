# v13: new engine, and a different way of choosing what it runs

New files throughout (`engine/engine.py`, `engine/model.py`, `engine/planner.py`,
`engine/kernels/{gemm,attention,elementwise,reference}.py`); the v11/v12 engine
and its kernels are gone from `engine/`. The `fast-engine` branch is not a
parent of any of this.

Three changes carry it, and only the third is a new idea about kernels. The
first two are about the method.

## 1. The measurement that was never taken

Every engine from v1 to v12 chose its tiles the same way: a hand-written
shortlist per projection, compiled and raced at warmup, keep the winner. Two
costs fall out of that, and both showed up in the run reports.

Compiling ten specialisations of five roles is most of the warmup budget, so
the list had to stay short. And a race only decides between what is on the
list, so **the list is the search space**. v11 widened it, spent the whole
budget, and came back with a batch-16 TPOT of 4.803 ms against v8's 4.808 --
identical on the shape the design targeted. v12 then diagnosed why (the
activation re-read, below) and fixed the ordering, but kept the same method.

v13 inverts it. `engine/planner.py` generates the candidate space from the
device and the role's own dimensions, scores every point on modelled traffic,
and the tuner compiles only the handful the model cannot separate. The model's
three constants are **measured on the machine at warmup**, not assumed:

| constant | how it is measured |
|---|---|
| `bw` | achieved bytes per second, from a real projection over all 36 layers |
| `launch_s` | the slope of two graphs that differ only in how many kernels they hold |
| `l2_cost` | solved from the same projection at two column widths |

The last one is the interesting one. The two tiles stream identical weights
and run the same grid, so the weight term and the wave efficiency cancel, and
what separates them is the activation re-read -- 31.5 MB against 3.9 MB for
the QKV projection. Writing both measurements as the model computes them and
dividing gives one equation in one unknown:

```
c = W (1 - R) / (R * l_wide - l_narrow),    R = busy_narrow * e_n / busy_wide * e_w
```

`tests/test_planner.py` checks this by planting constants, generating the two
times from the model, and requiring the solver to give them back.

The point is not that the model is right. It is that the shortlist is now
derived from a measurement of *this* machine rather than from a guess made on
a different one, and that the budget goes to measuring the machine instead of
compiling nine tiles that were never going to win. Measurement still decides:
every shortlisted tile is timed over all 36 layers' own weights, minimum of
rounds, and the tile v1 through v12 all ran (`16x256`) is offered first and
has to be beaten by 0.5% to be displaced.

## 2. What the model prices

Straight from v12's diagnosis, now written down as arithmetic rather than as a
ranking heuristic. A projection launches `N / BLOCK_N` programs and **each one
streams the whole `[M, K]` input tile**:

| role | N | K | weights | x re-reads at `BLOCK_N=16` |
|---|---:|---:|---:|---:|
| qkv | 6144 | 2560 | 31.5 MB | 31.5 MB |
| o | 2560 | 4096 | 21.0 MB | 21.0 MB |
| gate/up | 9728 | 2560 | 99.6 MB | 49.8 MB |
| down | 2560 | 9728 | 49.8 MB | 49.8 MB |
| lm | 151936 | 2560 | 777.9 MB | 777.9 MB |

8.05 GB of weights delivered by 14.30 GB of loads. Widening the tile divides
the second column and empties the device; splitting K fills it back up without
adding any x traffic, because each slice program reads only `K / split` of the
row. Neither works alone, which is why v4 (split-K alone, +1.3%) and v11 (wide
tiles alone, flat) both stalled.

The model also charges two things v12's did not:

* **The RMSNorm hand-off.** A residual epilogue leaves one partial sum of
  squares per column tile and *every* program of the next projection reads all
  of them. A producer using a 16-wide tile leaves 160 partials and bills its
  consumer 3.9 MB per launch for them. Wide producer tiles are now paid for
  twice, which is correct.
* **Starvation**, as a multiplier: too few bytes in flight per SM, and a K
  slice with fewer iterations than the pipeline has stages. Without these the
  model recommends the tile that moves the fewest bytes and then waits for all
  of them.

At the fallback constants the model puts a batch-16 step at 4.20 ms against
6.14 ms for the incumbent tiles, and the LM head is the single largest item in
that difference: 778 MB of x re-reads at `BLOCK_N=16`, 97 MB at 128, for a
projection that needs no split-K at all to keep the device full. Treat the
numbers as an ordering, not a forecast.

## 3. Split-K without the second launch

This is the new kernel idea, and it is what makes the wide-tile/split-K pair
affordable everywhere rather than only where it was worth a launch.

Splitting K has always needed a reduction, and the reduction has always been a
second kernel. At the ~2.8 us a launch costs in this sandbox and 36 layers,
that is 0.1 ms of a 4.8 ms step for each role that splits -- v6 measured
exactly this, in reverse, when dropping 36 launches was worth 2% at batch 16.
Three of five roles splitting is 0.3 ms, which is most of what the split was
going to win.

So the reduction happens inside the same launch, in the CUTLASS "last block
does the fixup" pattern that `cutlass`'s stream-K and the W4A16 split-K work
both use. Each program stores its FP32 partial, syncs its block, and bumps an
atomic counter for its column tile; the program that reads back `SPLIT - 1`
knows every slice is published, sums them **in slice order** -- so the result
is deterministic, not an atomic free-for-all -- and runs the epilogue. Nothing
spins, so nothing can deadlock on a block that was never scheduled, which is
the failure mode the optimization guide warns about for megakernels.

Two details make it safe to ship:

* The ordering is the documented release/acquire pattern, not a hope.
  `tl.debug_barrier()` puts every thread's store ahead of the counter, and
  Triton's scalar `tl.atomic_add` lowers to `atom.acq_rel.gpu` issued by one
  thread with the result broadcast through shared memory (verified against
  Triton 3.1's `LoadStoreOpToLLVM.cpp`), so the release covers the whole
  block's stores. Partials are stored and re-read with `.cg`, which keeps them
  out of the non-coherent L1.
* It is checked at load time against the two-launch path on a grid of hundreds
  of programs, repeated, with the counters required to come back at zero --
  and if it disagrees even once, the engine plans without it and the reduce
  kernel does the work. The same counter pattern merges the decode attention's
  split softmaxes, so attention can now be split for parallelism at batch 1
  and still cost one launch, instead of choosing between them as v6 had to.

## What else changed

* **Attention** keeps the fused head-norm/RoPE/cache-write and gains the
  in-kernel merge above; warps follow the KV block width instead of being
  pinned at 4.
* **Prefill** runs in row chunks, so the MLP's activations stop scaling with
  `batch * prompt`, and is pinned to FlashAttention when a probe in the
  engine's real layouts says the backend will take it -- an SDPA call the
  dispatcher cannot serve silently materialises a `[B, 32, T, T]` score matrix,
  a gigabyte at batch 4 over 2048 tokens. If the pinned call is refused at run
  time the engine unpins itself and carries on.
* **Every fast path has a floor.** Each kernel is checked against a PyTorch
  twin at load and replaced by the twin if it disagrees; the fused step is
  checked against a cuBLAS step at the real shape and then has to win a timed
  race against it; the twin attention reads its length from the device, so
  even with no Triton at all the step is still capturable as a CUDA graph.

## What is checked without a GPU

There is no GPU here, so the usual answer -- "it passed a public run" -- is not
available, and the platform has retired public runs in any case. Three suites
run offline, 33 tests:

* `tests/test_planner.py` -- tile legality, partial-buffer and hand-off
  bounds, shortlist ordering, and the calibration round-trip.
* `tests/test_kernels_sim.py` -- the kernels' **own source** executed over
  numpy by `tests/tlsim.py`, one program at a time, against the PyTorch twins.
  Triton 3.1's own interpreter reads and writes through device pointers and
  returns garbage for CPU tensors, so this is the only way to run them here.
  Every masked load asserts it is inside its buffer, and `bfloat16` is modelled
  as float32 rounded after each cast -- which is enough to tell the reference
  RMSNorm from the tidier formulation the contract forbids, and the test
  asserts it can.
* `tests/test_engine_cpu.py` -- the whole engine against Transformers' Qwen3
  on a tiny random model: identical tokens at three shapes, state reset across
  calls, a judge-style teacher-forced replay, the generator contract, and the
  fused step against the reference step at batch 1, 2 and 3.

That last suite already earned its keep: it caught the buffers being allocated
inside `torch.inference_mode()`, which makes them inference tensors and makes
every later in-place warmup update illegal.

What none of it covers: occupancy, timing, and concurrency. The fixup's
cross-block ordering runs sequentially here and can only be tested on the
device, which is what the load-time check is for.

## Considered and rejected

* **A blocked `[N/bn, K/bk, bn, bk]` weight layout.** At `BLOCK_K >= 64` every
  row segment is already 128 contiguous bytes -- a full L2 line -- so the
  layout buys DRAM page locality at best, and it needs a second copy of all
  8 GB because prefill hands the same matrices to cuBLAS.
* **A decode megakernel.** Worth ~0.4 ms of launch and ramp, and there is no
  grid-wide barrier for an ordinary launch. The fixup above takes the part of
  it that does not need one.
* **Fusing the QKV projection into attention.** It costs no extra weight
  traffic -- a program owning KV head `g` needs exactly the 768 columns of the
  projection that feed it -- and saves 36 launches. But it fixes the attention
  decomposition to `batch * 8` programs, which is 8 programs at batch 1, and
  it cannot then split the KV. Shape-dependent in a way the hidden set would
  punish.
* **Pruning the LM head with certified bounds.** Exactly computing
  `argmax(W h)` while skipping column tiles that provably cannot contain the
  maximum, using per-column `||w_j - dequant(w_j)||` from an INT8 copy. The
  arithmetic is sound and the pruning is exact, but survivors scatter across
  the vocabulary, so at 64-column granularity almost no tile is fully pruned,
  and the INT8 pass costs 194 MB to find that out.
* **Quantising the weights.** Ruled out permanently in v11's notes and not
  revisited: the 2.0-logit margin was calibrated on BF16 reduction-order
  noise, not on representation error.

## What would come next

1. **Read the run.** One change per run, the three public shapes' p50, TTFT
   and TPOT, and nothing below 2% believed. If the LM head and the two
   `N = hidden` projections move together, the traffic model is right.
2. **Speculation with a draft worth verifying.** Still the only lever with a
   factor in it -- decode is bandwidth-bound and speculation is the only thing
   that changes tokens per weight-stream. v10 built the exact verifier and
   failed on the *draft source*: n-grams have nothing to match in a prompt of
   random token ids. A directly-cast INT8 or MXFP4 copy of the same weights is
   a draft that costs a quarter of a step and agrees with the target most of
   the time (ML-SpecQD reports up to 2x from exactly this on memory-bound
   decode), and the contract is explicit that exact verification passes by
   construction. The risk is the 25% sample-spread gate, not correctness.
3. **A persistent grid**, once the traffic model is confirmed: it would make
   wave quantisation irrelevant and let `BLOCK_N` be chosen on traffic alone.
