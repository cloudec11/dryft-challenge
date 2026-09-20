# Agent briefing: Dryft "Decode Qwen3 4B faster"

A condensed, self-contained digest of this repository's README, AGENTS.md,
QWEN_ENGINE_CONTRACT.md, OPTIMIZATION_GUIDE.md, agent/README.md, and source.
Read it fully before editing. When in doubt, the contract
(`QWEN_ENGINE_CONTRACT.md`, or https://htn.dryft.ai/docs) wins.

---

## 1. Objective

Build the fastest inference engine for greedy decoding of
`Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, on **one H100**, such that
**every emitted token is what native Qwen would pick**.

- **Score** = geometric mean of output tokens/sec across **three hidden
  workloads** (equal weight). Throughput = `batch × output_tokens / median
  generation seconds`, **including prefill**. Higher is better.
- Native timings set the latency gates; they are not the score.
- Only `engine/` is submitted. Everything else (agent code, notes, logs,
  credentials) stays outside it.

## 2. Hard invariants (break one → run fails; speed cannot compensate)

1. `engine/engine.py` at the archive root exports `class Engine` with exactly:
   - `__init__(self, model_path: str) -> None` — load the checkpoint from
     `model_path` (a Hugging Face–layout directory). Untimed, but budgeted.
   - `generate(self, input_ids: list[list[int]], max_new_tokens: int)` — a
     **generator**. Yields one `list[int]` per step (one token per sequence, in
     input order), **exactly `max_new_tokens` times**. All sequences have equal
     length (no padding). **Never stop at EOS**; it is an ordinary id.
   - No manifest, no config, no other entry point, no kernel registry.
2. **Correctness:** after your engine dies, the judge replays your token sequence
   through native Qwen, teacher-forced on *your* prefix. At every output
   position your token must be the native argmax **or within 2.0 logits of
   it** (`tieMarginLogits`). One bad position fails the workload. Because replay
   follows your prefix, a legitimate near-tie flip does not cascade.
3. **No network in the container.** No pip, no downloads. Vendor pure-Python or
   Triton source inside `engine/`.
4. `engine/` contains only what the engine imports.
5. **Stream, don't batch the output.** Computing the whole continuation before
   the first yield destroys TTFT. Keep `.tolist()`/`yield` outside any CUDA
   graph capture.

## 3. Gates (any one fails the workload, and with it the run)

| Gate | Limit |
| --- | --- |
| Wrong token | any position outside the 2.0-logit tie margin |
| Time to first token (TTFT) | > 1.10× native median (enforced on official runs only) |
| Time per output token (TPOT) | > 1.10× native median (enforced on official runs only) |
| Peak GPU memory | > 90% of the device |
| Sample spread | > 25% across the five official samples |
| Load + one warmup | > 300 s |
| One sample | > 300 s |
| Crash / wrong step or token count | fail |

Traps:
- A change that trades latency for throughput can raise tok/s but fail the
  TTFT or TPOT gate.
- Anything whose cost varies by prompt (recompiles, dynamic-shape autotuning,
  heuristic slow paths) can pass a one-sample public run and then fail the 25%
  spread gate on the official run.
- Public runs **report** the TTFT and TPOT ratios but don't enforce them. Check
  them yourself before requesting an official run.

Failure codes that mean your engine is at fault: `incorrect_output`,
`candidate_error`, `timeout`, `latency_limit`, `memory_limit`, `unstable_timing`.
The platform is at fault for `harness_error` and infrastructure codes; retry
those without changing code. The log page shows a bounded tail of your engine's
stdout/stderr, so log enough there to diagnose a failure.

## 4. Workloads and timing

| Workload | batch | prompt | output | scored |
| --- | ---: | ---: | ---: | --- |
| public-0 | 1 | 512 | 32 | no |
| public-1 | 4 | 2048 | 32 | no |
| public-2 | 16 | 512 | 128 | no |
| hidden ×3 | ? | ? | ? | **yes** |

**Don't hard-code shapes.** The hidden workloads' shapes are unknown.

Per workload:
1. Fresh process.
2. `Engine(model_path)`.
3. One warmup `generate` of the **same shape**.
4. The samples: 1 for a public run, 5 for an official run.

Load and warmup share one 300 s budget and are untimed. Each sample's prompt is
freshly random token ids. You never see text, and you never see the same prompt
twice.

- TTFT = time until the first yielded step reaches the timing process.
- TPOT = (total − TTFT) / (steps − 1).
- Native is measured through the same pipe, in the same container. The order
  (native first or engine first) alternates per workload. Clocks are not locked.

Implications:
- Do weight relayout and packing in `__init__`.
- Allocate shape-dependent buffers, autotune, compile every Triton
  specialization, and capture CUDA graphs **during warmup**, when the shapes are
  known.
- **Reset all prompt-dependent state at the start of every `generate` call.**
  Warmup and sample prompts must never share cache contents.

## 5. Runtime (pinned; matches `requirements.txt`)

Python 3.11, CUDA 12.4, PyTorch 2.5.1, **Triton 3.1.0** (check APIs against
3.1.0, not newer releases), Transformers 4.51.3, safetensors 0.5.3,
tokenizers 0.21.1.

The engine runs as an unprivileged user with read-only access to the
checkpoint. The submission root is on `sys.path`, so a module beside
`engine.py` is imported by name: `import planner`, `from kernels import gemm`.

**Packaging limits:**
- 2 MiB compressed, 16 MiB expanded, 200 files.
- Allowed extensions: `.py .pyi .yaml .yml .json .toml .txt .md .cfg .ini`.
- Not allowed: `.cu`/`.cuh`, `.so`, cubin, PTX, weights, or binaries. Ship
  source only; Triton compiles at runtime.
- Archive paths must be `engine.py`, not `./engine.py`.

## 6. Model facts (Qwen3-4B, Transformers 4.51.3)

| Property | Value |
| --- | ---: |
| Decoder layers | 36 |
| Hidden `H` | 2560 |
| MLP intermediate `I` | 9728 |
| Vocab `V` | 151,936 |
| Query heads / KV heads / head dim | 32 / 8 / 128 (GQA, 4 q-heads per kv-head; q-head `h` → kv-head `h // 4`) |
| Activation | SwiGLU: `down(silu(gate(x)) * up(x))` |
| Norm | RMSNorm, eps 1e-6 |
| RoPE | theta 5,000,000, absolute positions |
| Biases | none |
| Sliding window | none |
| Embedding / LM head | tied |

`H ≠ Nq·D`: `q_proj` maps 2560 → 4096, and `o_proj` maps 4096 → 2560.

Per-layer forward:
```text
n = input_layernorm(x)
q, k, v = q_proj(n), k_proj(n), v_proj(n)
q, k = q_norm(q), k_norm(k)            # per-128-dim-head RMSNorm, BEFORE RoPE; V not normed
q, k = rope(q, k, absolute_positions)
k, v = update_layer_cache(k, v)
a = x + o_proj(gqa_attention(q, k, v)) # scale 1/sqrt(128), causal
m = post_attention_layernorm(a)
y = a + down_proj(silu(gate_proj(m)) * up_proj(m))
```
The full model is `embed → 36 layers → final norm → last position → LM head → argmax`.

Weight shapes (`[out, in]`, so `linear(x) = x @ W.T`):

| Weight | Shape |
| --- | --- |
| `embed_tokens`, `lm_head` (tied) | `[151936, 2560]` |
| `q_proj` | `[4096, 2560]` |
| `k_proj`, `v_proj` | `[1024, 2560]` each |
| `o_proj` | `[2560, 4096]` |
| `gate_proj`, `up_proj` | `[9728, 2560]` each |
| `down_proj` | `[2560, 9728]` |
| `input_layernorm`, `post_attention_layernorm`, `norm` | `[2560]` |
| `q_norm`, `k_norm` | `[128]` each |

Access weights through the loaded modules (`base = model.model`,
`base.layers[i]`), not the safetensors shards, so the embedding and LM head stay
tied.

KV cache size: `36 × 2 × B × seq × 8 × 128 × 2 bytes = 147,456 × B × seq` bytes,
or 144 KiB per token per sequence.

## 7. Numerics: the 2.0-logit budget

Native Qwen measured against itself (cached decode vs. one full forward) drifts
up to about 0.75 logits at a few positions per 12k, purely from BF16 summation
order.

- **Allowed:** reordering reductions; accumulating in FP32 where the reference
  accumulates in FP32.
- **Not allowed:** reformulating the math, including moving a BF16 cast
  boundary. The canonical example is RMSNorm (`engine/kernels/rmsnorm.py`):
  reduce and normalize in FP32, **cast to BF16, then multiply by the weight**.
  Multiplying in FP32 and rounding once at the end is more accurate, but it is
  a different function and can exceed the margin.
- **Never use:** quantization, KV eviction or pruning, approximate or sparse
  attention, or an unverified draft model. Each shifts logits by whole units.
- The argmax must break exact ties by taking the **lowest index**, as
  `torch.argmax` does.
- The baseline sets `allow_tf32 = False` for matmul and cuDNN. Keep that.

## 8. Repository layout

| Path | Submitted | Notes |
| --- | --- | --- |
| `engine/engine.py` | yes | The engine: load-time kernel checks, warmup calibration and tuning, CUDA-graphed decode, chunked prefill, streaming `generate`. |
| `engine/model.py` | yes | Checkpoint load, weight packing (`qkv`, `gate_up`), RoPE tables, per-role dimensions. |
| `engine/planner.py` | yes | The decode plan: device constants, the traffic model, and the tile candidates. No torch, no triton — testable without a GPU. |
| `engine/kernels/gemm.py` | yes | The projection kernel every decode matrix product runs on, with the elementwise work fused in and split-K reduced inside the same launch. |
| `engine/kernels/attention.py` | yes | Single-token GQA over the fixed-capacity KV cache, head norm + RoPE + cache write folded in, split softmaxes merged in the same launch. |
| `engine/kernels/elementwise.py` | yes | Embedding, prefill head norm + RoPE + cache write, SwiGLU, RMSNorm. |
| `engine/kernels/reference.py` | yes | PyTorch twins of every kernel. The load-time judge, and the fallback when one disagrees. |
| `agent/package.py` | no | Builds a reproducible tar.gz from `engine/` and enforces the platform limits (checks `engine.py` exists and defines `class Engine`). |
| `agent/client.py` | no | Standard-library API client `Dryft` with `benchmark()`, `submit(bytes)`, `start_run(id, mode)`, `run(id)`, `logs(id)`, and `wait(id)`. Needs `DRYFT_API` and `DRYFT_TOKEN`. |
| `agent/loop.py` | no | `attempt()` runs package → submit → run → `report()`, which prints per-workload speedup and TTFT/TPOT ratios and flags `OVER GATE` above 1.10. `plan_next_edit()` is an unimplemented stub for you to write. |
| `bin/` | no | Target for the Dryft CLI (`install-dryft.sh` or `install-dryft.ps1`). |
| `tests/test_planner.py` | no | The decode plan's invariants and the calibration round-trip. Runs anywhere. |
| `tests/test_kernels_sim.py` | no | The Triton kernels' own source, run over numpy by `tests/tlsim.py`, against the twins. Triton's own interpreter cannot do this: it returns garbage for CPU tensors. |
| `tests/test_engine_cpu.py` | no | The whole engine against Transformers' Qwen3 on a tiny model. Needs CPU torch. |
| `tests/test_client.py` | no | Unit tests for the client. |

Run the three engine suites with `python tests/<name>.py`; none of them needs a
GPU, pytest, or the checkpoint.

## 9. Submitting

The CLI knows the event server; it only needs `DRYFT_TOKEN` (create one at
https://htn.dryft.ai/tokens). Set `DRYFT_API` only for local, staging, or
self-hosted servers, with no `/api` suffix.

```sh
./install-dryft.sh            # or .\install-dryft.ps1 on Windows
export DRYFT_TOKEN='dryft_pat_...'
./bin/dryft doctor
./bin/dryft validate engine
./bin/dryft submit engine                              # prints submission id
./bin/dryft run <submission-id> --mode public --wait 3000
./bin/dryft run <submission-id> --mode official --wait 3000
./bin/dryft logs <run-id> --follow
./bin/dryft result <run-id> --wait --timeout 3000
```

A connected GitHub repository with **Engine folder** set to `engine` also runs
on every push to the default branch.

If a wait times out, **don't start a second run.** Keep the run id and poll
again.

Workflow:
1. Edit `engine/`.
2. Submit a **public** run (about 2 min, one sample, never ranked).
3. Read tok/s, TTFT, TPOT, memory, and the native ratios.
4. Once results are clean and the ratios have headroom, request an **official**
   run (5 samples, hidden shapes scored, spread gate live).

Record every attempt and its result outside `engine/`.

## 10. Optimization roadmap (rough payoff order for these shapes)

1. **Per-step overhead** (dominant at batch 1 and 4): skip the
   `Qwen3ForCausalLM` wrapper with a hand-rolled forward, then CUDA-graph the
   single-token decode step. Keep graph I/O addresses stable, update token ids
   and positions in place, and capture per shape during warmup.
2. **Static KV cache:** preallocate per-layer K and V as
   `[B, 8, prompt_len + max_new_tokens, 128]`. Write at `[:, :, L:L+T]`. CUDA
   graphs need fixed shapes, so attention must read a device-side length or use
   a mask over the fixed capacity, never a growing Python slice. Never read
   unused capacity.
3. **Prefill:** a large share of total time and all of TTFT (prompts of
   512–2048 tokens vs. outputs of 32–128). Use a separate prefill path with its
   own kernels or tiling. Consider chunked prefill or overlapping it with the
   first decode steps.
4. **Fused kernels:** RMSNorm; QK-norm plus RoPE (optionally writing K straight
   into the cache); a fused QKV projection with packed weights; SwiGLU; an o_proj
   and down_proj residual epilogue; a GQA attention kernel that maps q-head
   `h → h // 4` directly instead of expanding KV; LM head plus argmax.
   Preserve the reference BF16 rounding points in every one of them.
5. **Speculative decoding with exact verification:** legal, and correct by
   construction, but expensive to build. Do it last.
6. **Megakernels are allowed but hard.** There is no grid-wide barrier for
   ordinary launches, and spin-waiting on blocks that haven't been scheduled can
   deadlock. Prefer CUDA-graph replay of separate launches first.

Staged integration pattern (examples in `OPTIMIZATION_GUIDE.md`):
1. Swap one leaf module, such as a `FusedRMSNorm` adapter that reuses
   `.weight` and `.variance_epsilon`, keeping Transformers' cache and attention.
   Check it against the baseline.
2. Bypass the top-level wrapper by calling `base.embed_tokens`,
   `base.rotary_emb`, each layer with `DynamicCache`, `base.norm`, and
   `lm_head` on the last position. `attention_mask=None` is **only** valid for a
   full, unpadded prefill into an empty cache followed by T=1 decode.
3. For chunked prefill, multi-token verification, or a static cache, supply an
   explicit mask: key `j` is visible iff `j <= L + query_index` and the slot has
   been written.
4. Replace the cache, capture decode in a CUDA graph, then replace full blocks.

When replacing a Transformers module, keep its call interface. Norms and MLPs
return a tensor. `self_attn.forward` returns `(attn_output, attn_weights_or_None)`.
Replacing attention also makes your code responsible for cache updates.

## 11. Local verification checklist (if a GPU is available)

Compare against an untouched baseline under `torch.inference_mode()`:
- prefill logits;
- several cached decode steps;
- two consecutive `generate` calls with different prompts, to prove state is
  reset;
- all three public shapes, plus a few other shapes (the hidden workloads are
  unknown);
- the maximum logit delta at the chosen token, which must stay well under 2.0
  (aim to stay near the 0.75 noise floor);
- TTFT, TPOT, throughput, and peak memory.

The platform's teacher-forced replay is the final judge.
