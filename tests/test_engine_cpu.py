"""CPU check of engine logic against Transformers' Qwen3 on a tiny random model.

Needs CPU PyTorch and transformers==4.51.3; no GPU. Compares the engine's
tokens with the starter's Transformers loop and runs a judge-style replay.

    python tests/test_engine_cpu.py                     # PyTorch kernels, FP32 + BF16
    TRITON_INTERPRET=1 python tests/test_engine_cpu.py  # Triton kernels, FP32
    TRITON_INTERPRET=1 FORCE_TRITON=1 python tests/test_engine_cpu.py

The interpreter has no BF16, so BF16 rounding inside Triton kernels is only
checked on the GPU, by the engine's load-time self-checks.
"""
import copy
import os
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine  # noqa: E402

torch.manual_seed(0)


def tiny_model(dtype):
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=256, intermediate_size=640, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, head_dim=128, rope_theta=5e6,
        rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=4096,
        initializer_range=0.15, attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
    return model.to(dtype)


def baseline_generate(model, input_ids, n):
    """The starter engine's loop, verbatim."""
    current = torch.tensor(input_ids, dtype=torch.int64)
    cache = None
    out = []
    with torch.inference_mode():
        for _ in range(n):
            o = model(input_ids=current, past_key_values=cache, use_cache=True,
                      logits_to_keep=1, return_dict=True)
            current = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache = o.past_key_values
            out.append(current[:, 0].tolist())
    return out


def replay_margin(model, input_ids, steps):
    """Judge-style check: max over positions of (argmax logit - our logit)."""
    ids = torch.tensor(input_ids)
    gen = torch.tensor(steps).T  # [B, N]
    full = torch.cat([ids, gen], dim=1)
    with torch.inference_mode():
        logits = model(input_ids=full).logits.float()
    S = ids.shape[1]
    pred = logits[:, S - 1:-1, :]  # predicts each generated token
    chosen = pred.gather(-1, gen[..., None])[..., 0]
    return (pred.max(-1).values - chosen).max().item()


def run(dtype, use_triton, shapes):
    model = tiny_model(dtype)
    ref = copy.deepcopy(model)
    eng = Engine.from_model(model, "cpu", use_triton=use_triton, use_graphs=False)
    if os.environ.get("FORCE_TRITON") == "1":
        from kernels import misc
        eng.k_add_norm, eng.k_qkv_post = misc.add_rms_norm, misc.qkv_post
        eng.k_silu_mul, eng.k_embed = misc.silu_mul, misc.embed
        eng.use_triton_attn = True
    ok = True
    for (B, S, N) in shapes:
        for trial in range(2):  # two calls per shape: state must reset
            prompt = torch.randint(0, 512, (B, S)).tolist()
            got = list(eng.generate(prompt, N))
            assert len(got) == N and all(len(s) == B for s in got), "bad step shape"
            want = baseline_generate(ref, prompt, N)
            margin = replay_margin(ref, prompt, got)
            same = got == want
            flag = "ok" if (margin <= 2.0 and (same or dtype != torch.float32)) else "FAIL"
            ok &= flag == "ok"
            print(f"  {str(dtype):14} triton={use_triton!s:5} B={B} S={S} N={N} "
                  f"#{trial}: identical={same} replay_margin={margin:.4f} {flag}")
    return ok


shapes = [(1, 17, 6), (3, 40, 9), (2, 64, 1), (4, 130, 12)]
results = []
if os.environ.get("TRITON_INTERPRET") == "1":
    results.append(run(torch.float32, True, shapes))
else:
    results.append(run(torch.float32, False, shapes))
    results.append(run(torch.bfloat16, False, shapes))
print("ALL OK" if all(results) else "SOME FAILED")
