# Attempts log

One entry per platform run. Score = geometric mean tok/s over the 6 hidden
workloads. Fill in the numbers from the run page.

| # | Commit | Change | Score (tok/s) | Pass? | Notes |
|---|--------|--------|---------------|-------|-------|
| 0 | 186912a | Baseline (Transformers, unchanged) | _hidden score?_ | yes | samples below |
| 1 | 5aed25f | v1: hand-rolled forward, static KV cache, CUDA-graph decode, fused Triton kernels | 855.3 | yes | official run c0c37a87, leaderboard #25; samples below |
| 2 | _tbd_ | v2: fused skinny-GEMM decode step (6 launches/layer), GQA flash prefill, row qkv_post, 3-step lookahead | | | |

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
