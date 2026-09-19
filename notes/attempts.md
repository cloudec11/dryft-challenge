# Attempts log

One entry per platform run. Score = geometric mean tok/s over the 6 hidden
workloads. Fill in the numbers from the run page.

| # | Commit | Change | Score (tok/s) | Pass? | Notes |
|---|--------|--------|---------------|-------|-------|
| 0 | 186912a | Baseline (Transformers, unchanged) | _hidden score?_ | yes | samples below |
| 1 | _tbd_ | v1: hand-rolled forward, static KV cache, CUDA-graph decode, fused Triton kernels | | | |

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
