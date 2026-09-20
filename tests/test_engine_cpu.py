"""End-to-end check of the engine against Transformers' own Qwen3, on CPU.

This is the test that would have caught every wiring bug the engine can have:
a residual added into the wrong buffer, a hand-off read with the wrong stride,
a cache slot written one position late, state surviving from the previous
call.  None of those need a GPU to find.  What it cannot check is the Triton
kernels' BF16 rounding, which only the engine's own load-time checks see.

It runs a tiny random Qwen3 -- three layers, 512 tokens of vocabulary -- so
the whole file takes a few seconds:

    python tests/test_engine_cpu.py

Needs CPU PyTorch and transformers 4.51.3.  Skips itself if they are missing.
"""

import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError as exc:  # pragma: no cover - offline machines
    print(f"skipped: {exc}")
    raise SystemExit(0)

import planner  # noqa: E402
from engine import Engine  # noqa: E402

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
    return condition


def tiny_model(dtype=torch.float32, layers=3):
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=256, intermediate_size=640,
        num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=2,
        head_dim=64, rope_theta=5e6, rms_norm_eps=1e-6, tie_word_embeddings=True,
        max_position_embeddings=4096, attn_implementation="sdpa",
        use_sliding_window=False, sliding_window=None,
    )
    model = Qwen3ForCausalLM(cfg).to(dtype).eval()
    # Random init leaves the norms at 1.0, which hides a gain that is read
    # from the wrong layer.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.normal_(1.0, 0.05)
    return model


def baseline_tokens(model, prompts, steps):
    """The starter engine's loop: Transformers, greedy, with a KV cache."""
    ids = torch.tensor(prompts, dtype=torch.int64)
    out = []
    with torch.inference_mode():
        past, current = None, ids
        for _ in range(steps):
            result = model(input_ids=current, past_key_values=past, use_cache=True,
                           logits_to_keep=1)
            past = result.past_key_values
            current = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(current[:, 0].tolist())
    return out


def engine_tokens(engine, prompts, steps):
    return [row for row in engine.generate(prompts, steps)]


def test_matches_transformers():
    """Same tokens as the reference loop, at three shapes."""
    model = tiny_model()
    for batch, prompt_len, steps in ((1, 24, 6), (3, 16, 5), (2, 9, 8)):
        torch.manual_seed(batch * 100 + prompt_len)
        prompts = torch.randint(0, 512, (batch, prompt_len)).tolist()
        want = baseline_tokens(model, prompts, steps)
        engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                                   use_graphs=False)
        got = engine_tokens(engine, prompts, steps)
        check(len(got) == steps, f"B={batch} S={prompt_len}: {len(got)} steps, "
                                 f"wanted exactly {steps}")
        check(all(len(row) == batch for row in got),
              f"B={batch}: a step yielded the wrong number of tokens")
        check(got == want, f"B={batch} S={prompt_len}: tokens differ\n"
                           f"  got  {got}\n  want {want}")


def test_state_is_reset_between_calls():
    """Two calls of the same shape, then the same prompts in the other order:
    anything left in the cache from the first call shows up here."""
    model = tiny_model()
    torch.manual_seed(7)
    first = torch.randint(0, 512, (2, 12)).tolist()
    second = torch.randint(0, 512, (2, 12)).tolist()
    engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                               use_graphs=False)
    got_first = engine_tokens(engine, first, 4)
    got_second = engine_tokens(engine, second, 4)
    got_first_again = engine_tokens(engine, first, 4)
    check(got_first == got_first_again,
          f"the same prompt gave different tokens on the second visit:\n"
          f"  {got_first}\n  {got_first_again}")
    check(got_second == baseline_tokens(model, second, 4),
          "the second call did not match the reference loop")


def test_judge_replay_margin():
    """The judge's own test: replay the emitted tokens teacher-forced and
    check each one is still the argmax of the reference's logits."""
    model = tiny_model()
    torch.manual_seed(3)
    prompts = torch.randint(0, 512, (2, 20)).tolist()
    engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                               use_graphs=False)
    steps = 6
    emitted = engine_tokens(engine, prompts, steps)
    full = [list(p) + [row[b] for row in emitted] for b, p in enumerate(prompts)]
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor(full)).logits
    prompt_len = len(prompts[0])
    worst = 0.0
    for step in range(steps):
        column = logits[:, prompt_len - 1 + step, :]
        best = column.max(dim=-1).values
        for b in range(len(prompts)):
            chosen = column[b, emitted[step][b]]
            worst = max(worst, float(best[b] - chosen))
    check(worst < 2.0, f"replay margin {worst:.3f} is outside the 2.0 tie margin")
    check(worst < 1e-3, f"replay margin {worst:.3f}: FP32 should be exact")


def test_fused_step_matches_the_reference_step():
    """The fused step's bookkeeping, with the reference projection standing in
    for the Triton one.

    What this exercises is everything around the kernel: the two
    sum-of-squares buffers alternating between producer and consumer, the
    partial count each RMSNorm prologue is told to read, the residual updated
    in place by two different epilogues, and the tile the LM head is given.
    A kernel can be perfect and the step still wrong if any of that is.
    """
    model = tiny_model(layers=2)
    for batch, prompt_len, steps in ((2, 10, 4), (1, 7, 3), (3, 6, 2)):
        engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                                   use_graphs=False)
        torch.manual_seed(11 + batch)
        prompts = torch.randint(0, 512, (batch, prompt_len)).tolist()
        engine_tokens(engine, prompts, steps)  # builds the plan, fills the cache
        st = engine.state
        # A plan the reference projection can honour: the tiny model's
        # dimensions are not the checkpoint's, so the tiles are chosen for it
        # here rather than by the tuner, which needs a device.
        for role in ("qkv", "o", "gate_up", "down", "lm"):
            n, k, _, glu, _ = engine.w.role_dims(role)
            st.tiles[role] = planner.candidates(
                batch, n, k, glu, engine.dev, limit=1, allow_split=role != "lm",
                allow_fixup=False,
                max_parts=256 if role in ("o", "down") else None)[0]
        check(engine._check_fast(st),
              f"B={batch}: the fused step disagrees with the reference step")


def test_plan_survives_an_exhausted_budget():
    """Load plus warmup share 300 seconds, and going over is ``timeout`` --
    the run, not just the tuning.  So when the budget is gone the engine has
    to fall back to a plan it can trust without measuring, and that plan has
    to be legal and produce the same logits as the reference step."""
    model = tiny_model(layers=2)
    engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                               use_graphs=False)
    batch, prompt_len, steps = 2, 10, 3
    torch.manual_seed(21)
    prompts = torch.randint(0, 512, (batch, prompt_len)).tolist()
    engine_tokens(engine, prompts, steps)
    st = engine.state
    st.tiles.clear()
    engine._fill_incumbent_tiles(st)
    check(len(st.tiles) == 5, f"only {len(st.tiles)} of 5 roles were planned")
    for role, tile in st.tiles.items():
        n, k, _, _, _ = engine.w.role_dims(role)
        check(n % tile.bn == 0 and k % tile.bk == 0,
              f"fallback tile {tile} does not divide {role} (n={n} k={k})")
        if role in ("o", "down"):
            check(-(-n // tile.bn) <= 256,
                  f"fallback tile {tile} needs more hand-off rows than there are")
    check(engine._check_fast(st), "the fallback plan disagrees with the reference")
    # The budget itself has to be a real clock, counted from __init__.
    check(engine._remaining() < 255.0, "the budget did not start counting")


def test_generator_contract():
    """``generate`` is a generator that yields exactly ``max_new_tokens``
    lists, and yields nothing at all for a non-positive count."""
    model = tiny_model(layers=2)
    engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                               use_graphs=False)
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]
    for steps in (1, 2, 5):
        rows = list(engine.generate(prompts, steps))
        check(len(rows) == steps, f"{steps} asked for, {len(rows)} yielded")
        check(all(isinstance(r, list) and all(isinstance(t, int) for t in r)
                  for r in rows), "a step yielded something other than ints")
    check(list(engine.generate(prompts, 0)) == [], "zero steps yielded something")


def test_capacity_covers_every_write():
    """Attention writes cache slot ``pos`` on every step, including the ones
    warmup runs past the end of the generation."""
    model = tiny_model(layers=1)
    engine = Engine.from_model(copy.deepcopy(model), "cpu", use_triton=False,
                               use_graphs=False)
    for batch, prompt_len, steps in ((1, 4, 1), (2, 8, 3), (1, 5, 40)):
        st = engine._state_for(batch, prompt_len, steps)
        last_write = prompt_len + steps - 2
        check(st.capacity > last_write,
              f"B={batch} S={prompt_len} N={steps}: capacity {st.capacity} "
              f"does not cover slot {last_write}")
        check(st.capacity >= prompt_len + steps,
              f"capacity {st.capacity} leaves no warmup room")
        check(st.k_cache.shape[3] == st.capacity, "the cache is not capacity deep")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        before = len(FAILURES)
        try:
            test()
            status = "ok" if len(FAILURES) == before else "FAIL"
        except Exception as exc:  # pragma: no cover - a raised test is a failure
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
            status = "FAIL"
        print(f"{status:>4}  {test.__name__}", flush=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for message in FAILURES[:20]:
            print(f"  - {message}")
        return 1
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
