# Dryft "Decode Qwen3 4B faster": findings

Written 2026-09-20, covering v11 and v12 and the run log they sit in.
`notes/attempts.md` is the per-run journal; this is what the journal adds up
to. The short version: **the measurement is noisier than every engine
difference we have shipped, and we have been ranking designs with it anyway.**

## 1. The task

Decode `Qwen/Qwen3-4B-Instruct-2507` (revision `cdbee75f...`), BF16, one
H100, greedy, maximising output tokens/sec over six hidden workloads plus
three public ones. `engine/engine.py` exports `Engine(model_path)` with a
`generate(input_ids, max_new_tokens)` generator that yields exactly
`max_new_tokens` lists of one token id per sequence and never stops at EOS.

Gates: every token must be the teacher-forced argmax or within 2.0 logits of
it; TTFT and TPOT within 1.10x native; peak memory under 90%; sample spread
under 25%; load and warmup under 300 s.

The three public shapes, recovered from the result JSON via
`p50 = TTFT + (N-1) * TPOT`:

| shape | batch | prompt | new tokens |
|---|---:|---:|---:|
| public-0 | 1 | 512 | 32 |
| public-1 | 4 | 2048 | 32 |
| public-2 | 16 | 512 | 128 |

## 2. What the engine does

Unchanged in shape since v6, and all of it still stands:

- Weights packed at load (`qkv [6144, 2560]`, `gate_up [19456, 2560]`), with
  HF modules dropped as each layer is packed, so peak load memory is one
  layer of duplication rather than a second copy of the model.
- Static KV cache sized to `prompt + new_tokens`, CUDA-graphed decode, and a
  prefill graph raced against eager and kept only if it wins.
- One Triton projection kernel for all five roles (`qkv`, `o`, `gate_up`,
  `down`, `lm`), specialised on `NORM` / `GLU` / `RES` / `SPLIT`. Every
  elementwise op rides as a prologue or an epilogue: RMSNorm reads the
  partial sums of squares the *previous* projection's epilogue left behind,
  SwiGLU and the residual add happen in the epilogue, and Q/K head-norm,
  RoPE and the KV write ride inside attention. **A layer is five launches.**
- Flash-decoding attention that skips its reduce kernel when a single split
  covers the sequence (36 fewer launches per step).
- Every Triton kernel has a PyTorch twin in `kernels/ref.py`, is checked
  against it at load, and is replaced by the twin if it disagrees or fails to
  compile. The fused step is checked against the reference step's logits
  before it is allowed to race.

That correctness scaffolding has done its job: **every ranked run has come
back `correct: true` on every shape.** Nothing below is a correctness
problem.

## 3. The central finding: the measurement cannot see our changes

On 2026-09-20 a **notes-only** commit (`8a11f242`, one markdown file) fired a
run. Its engine was byte-identical to v11's:

| | score | B1 TPOT | B4 TPOT | B16 TPOT |
|---|---:|---:|---:|---:|
| v11 `91b25bef` | **898.54** | 3.990 | 4.832 | 4.803 |
| v11 `7229cc00`, identical bytes | **880.78** | 4.064 | 4.937 | 4.893 |
| spread | **2.0%** | 1.9% | 2.2% | 1.9% |

Two runs of the same code, 2.0% apart. Now put that next to every run that
has landed on the plateau:

| run | commit | what it was | score |
|---|---|---|---:|
| `84daa14e` | `cd5ef7ce` | v8 | **902.88** |
| `be619c9a` | `2fde548a` | v10, speculative decoding | 900.88 |
| `91b25bef` | `09c3f866` | v11, wave-quantisation rewrite | 898.54 |
| `a93ebf2b` | `beb07324` | | 894.48 |
| `b54796c8` | `6505b800` | | 888.23 |
| `d6358959` | `b285c46e` | | 884.56 |
| `881e52fa` | `31212d07` | v9, split-K everywhere | 883.57 |
| `3bdf08de` | `94660b01` | | 882.68 |
| `b4494a7a` | `912d2baa` | | 882.24 |
| `4e9a638e` | `daedece5` | | 882.15 |
| `7229cc00` | `8a11f242` | **v11 again, same bytes** | 880.78 |
| `ce136f30` | `1206a63f` | v12, re-read model | 872.17 |
| `42b2ac2b` | `75595eb1` | v14 warmup change | 864.45 |
| `c0c37a87` | `5aed25fd` | | 855.34 |
| `6d4b46c6` | `211a1ea8` | v13 | 843.24 |
| `9b03adfd` | `186912a6` | native baseline | 169.83 |

Eleven runs spanning roughly eight materially different engines sit inside
**880.78 to 902.88, a 2.5% band** — and one engine on its own accounts for
2.0% of it. Batch-16 TPOT across all of them is 4.799 to 5.013 ms, 4.5%,
with the *best* single value (4.799) belonging to `b54796c8`, which scored
888.

**So essentially every design decision made between v6 and v12 was ranked on
a difference smaller than the instrument's error, with one sample each.**
That includes decisions recorded in `attempts.md` as wins and as losses. v10
"regressed 0.22%" and v11 "regressed 0.48%"; both statements are noise. v8
holds the best score, but on this evidence v8 being best is substantially
luck.

There is a hint of structure rather than pure randomness: `d6358959`
recorded public-2 at 731.20 ms and `ce136f30` at 731.63 ms, while
`91b25bef` and `be619c9a` both landed near 716.4 ms. That looks like two
machine populations about 2% apart rather than a continuum, which would make
a run closer to a coin flip between two clusters than a draw from a narrow
distribution. Either way the consequence is the same.

### The one signal that is not noise

Batch-1 TPOT has moved monotonically, and by far more than the noise band:

```
3.863  a93ebf2b          <- best ever
3.883  v8
3.918  v10
3.964  d6358959
3.990  v11
4.064  v11 repeat
4.203  v12
4.306  v14 warmup
4.409  v13
```

3.863 to 4.409 is **14.1%**, an order of magnitude outside the 1.9% seen on
identical bytes. Whatever has accumulated in the batch-1 path is real, and it
is the only per-shape number in the whole log that clears its own error bar.
It is the thing to bisect.

## 4. What the step actually moves

Useful independently of the noise problem, because it is arithmetic rather
than measurement.

Per decode step at batch 16 over 640 slots:

| | bytes | at 3.35 TB/s |
|---|---:|---:|
| weights (7.27 GB layers + 0.78 GB tied LM head) | 8.05 GB | 2.40 ms |
| KV read | ~1.40 GB | 0.42 ms |
| ~185 launches at a measured ~2.8 us | | 0.52 ms |
| **accounted** | 9.45 GB | **3.34 ms** |
| **measured** | | **~4.80 ms** |

The ~2.8 us per launch is not a guess: v6 removed 36 launches per step by
letting single-split attention skip its reduce, and batch time fell 13 ms
over 127 steps.

There is a second, larger traffic term that v11 and earlier did not model.
Each of a projection's `N / BLOCK_N` programs streams the **whole**
`[BLOCK_M, K]` input tile. At `BLOCK_M = BLOCK_N = 16` — the tile every
version from v1 to v11 ran — the x tile and the weight tile per K-iteration
are both exactly 8192 bytes:

| role | N | K | weights | x re-reads |
|---|---:|---:|---:|---:|
| qkv | 6144 | 2560 | 31.5 MB | 31.5 MB |
| o | 2560 | 4096 | 21.0 MB | 21.0 MB |
| gate/up | 9728 | 2560 | 99.6 MB | 49.8 MB |
| down | 2560 | 9728 | 49.8 MB | 49.8 MB |
| lm | 151936 | 2560 | 777.9 MB | 777.9 MB |
| **per step** | | | **8.05 GB** | **6.25 GB** |

So the step issues **14.30 GB of loads to deliver 8.05 GB**. That is worth
knowing, but see section 5: acting on it did not help.

## 5. Hypotheses tested and falsified

1. **Wave quantisation is the binding cost** (v11). The `o` and `down`
   projections tile 2560 columns into 160 programs on 132 SMs — 1.21 rounds
   of work in 2 rounds, 61% occupancy. v11 rebuilt tile selection around it,
   and B16 TPOT came back 4.803 against v8's 4.808. Real, but not binding.
2. **Split-K everywhere** (v9). Extended v4's split-K from `o` and `down` to
   the other three roles. No effect — those roles already had full grids, and
   split-K without a wider tile changes nothing else.
3. **Speculative decoding via n-gram drafts** (v10). Flat. Sample prompts are
   freshly random token ids, so a prompt-lookup draft has almost nothing to
   match; v10's own gate most likely turned it off.
4. **Activation re-reads are the missing time** (v12). Cutting them ~4x via
   wide tiles plus split-K was modelled at +31% at batch 16 and delivered
   nothing (872.17, or about -1% against the v11 repeat at 880.78). The bytes
   are real; the stall is not. They are L2 hits that already overlap the HBM
   stream, and **counting bytes is not the same as finding a bottleneck.**

5. **Launch count is the binding cost** (v15, runs 16-18). The one term with
   direct evidence behind it: v6 removed 36 launches per step and B16 batch
   time fell 12 ms over 127 steps, i.e. ~2.8 us per launch, so a step's 185
   launches are ~0.52 ms of 4.80 ms. `kernels/layer.py` runs a whole layer in
   one launch with grid-wide barriers, 41 per step instead of 185. Its
   arithmetic is verified offline against torch (`tests/test_layer_sim.py`,
   both attention modes, plus a test that the bounded spin escapes rather
   than hangs) - but on the device it never ran: three runs, two of which
   adopt it on the logit check alone, all returned the fused step's numbers
   inside 0.3%. Not falsified, **untested**: what is left is the barrier not
   holding on hardware or a launch-time resource failure, and a ranked run
   reports neither. The clearest case in this log of the feedback channel,
   rather than the idea, being the limit.

Ruled out by the rules rather than by measurement: **quantising the model
weights.** `CONTEXT.md:189` and `AGENTS.md:92` both say never, and
`QWEN_ENGINE_CONTRACT.md:117` gives the reason — whole-unit logit shifts, and
one failing position fails the workload. A quantised *draft* under exact
verification is a different thing and is permitted.

## 6. Why we cannot tell hypotheses apart

- **No local GPU.** Nothing here can run a Triton kernel; offline coverage is
  13 tests over tile-plan arithmetic and fused-step wiring.
- **No public runs on this deployment.** `POST /api/v1/submissions` answers
  405 with `Allow: GET`. Pushing `main` is the only channel, and it fires an
  official ranked run.
- **No engine stdout.** `GET /api/v1/runs/{id}/logs` on a ranked run returns
  five harness lines and an explicit refusal: output written by the
  submission is not shown, because a hidden case's dimensions could be
  encoded into arbitrary text. **v11's per-role GB/s instrumentation was
  therefore unreadable — complexity spent on stderr diagnostics is wasted.**
- So one hypothesis costs ~10 minutes and returns three numbers whose error
  bar is wider than any effect we have produced.

The leaderboard keeps each team's *best*, so a bad run costs only the run.
That is the one thing working in our favour, and it argues for repetition.

## 7. What to do next, in order

1. **Fix the instrument before trusting it again.** Everything above says
   single-sample comparison is invalid here. Concretely: re-run the same
   commit two or three times and compare *distributions*, not points. The v11
   repeat was an accident; it should be the method.
2. **Rent an H100 for an hour.** ~$2-3 on RunPod or Lambda. `ncu --set full`
   on the decode graph would have killed the v12 hypothesis in twenty minutes
   by showing those L2 hits were not stalling, instead of costing a run and a
   day. Every remaining question is a profiler question.
3. **Bisect batch-1 TPOT.** 3.863 to 4.409 is the only real signal in the
   log. Candidates: the v11 vector kernel (introduced exactly where the slide
   steepens), and warmup or tuning work that lands on the first sample.
4. **Then, and only then, structural work.** A persistent grid-strided kernel
   would make wave quantisation irrelevant; a whole-layer megakernel is worth
   ~0.4 ms/step (~8%) but needs a grid-wide barrier Triton lacks, and getting
   it wrong is a hang that costs the 300 s limit. Neither is worth attempting
   blind.

The gap to #1 (1280.44, against our 902.88 at rank 24 of 56) is +42%. Nothing
in the byte budget explains a gap that size as tile tuning, and four tuning
hypotheses are now falsified. It is either a profiler-level insight we cannot
currently reach, or speculation with a draft worth verifying — v10 already
built the hard half of that (exact verification through the fused step at
~1.1x a plain step); what it lacked was a draft.

## 7b. What the noise looked like when we finally repeated a commit twice

The v11 repeat was an accident. Since then it has been done on purpose: the
v8 engine, restored byte-for-byte, was measured a second time.

| | score | B1 TPOT | B4 TPOT | B16 TPOT | public-2 p50 |
|---|---:|---:|---:|---:|---:|
| v8 `84daa14e` | 902.88 | 3.883 | 4.807 | 4.808 | 716.4 ms |
| v8 again `80b753b8` | 883.34 | 3.977 | 4.942 | 4.890 | 731.6 ms |

2.2% apart on identical bytes, and public-2 p50 lands on one of two values,
~716 ms or ~731 ms, never between. So the two-population guess in section 3
holds up, and there is a practical rule in it: **read public-2 p50 first to
see which machine a run got, then compare TPOT within that cluster.** Runs
16-18 were read that way, which is why "no change" was readable at all.

## 8. Repo state, 2026-09-20

`origin/main` is at `0cec425` (v14), pushed by a **concurrent agent session**
that also produced v13 (`211a1ea8`, 843.24) and the v14 warmup change
(`75595eb1`, 864.45), with `0ba80f9d` still measuring. A session earlier
`git reset --hard` on this working tree and deleted untracked work. If two
agents keep sharing `D:\codingt\dryft-challenge` they will keep colliding;
they need separate worktrees.

Local branches from this session:

- `main` at `975d8b2` — v12 plus the attempts entry; behind `origin/main`.
- `revert-v12` at `b2c52e4` — returns `engine/` **byte-for-byte to v11**
  while keeping `notes/v12-design.md` and the attempts entry. Not pushed;
  pushing spends a run.

Given section 3, reverting v12 is no longer clearly correct: at 872.17
against a v11 that itself measured anywhere from 880.78 to 898.54, v12 is not
demonstrably worse than what it replaced. Both are inside the band. The
honest position is that we do not know, and one more single run will not tell
us.
