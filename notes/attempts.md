# Attempts log

One entry per platform run. Score = geometric mean tok/s over the 6 hidden
workloads. Fill in the numbers from the run page.

| # | Commit | Change | Score (tok/s) | Pass? | Notes |
|---|--------|--------|---------------|-------|-------|
| 0 | 186912a | Baseline (Transformers, unchanged) | _pending_ | | |
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
