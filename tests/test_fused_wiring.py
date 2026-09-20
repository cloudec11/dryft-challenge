"""CPU check of the fused decode step's plumbing. No GPU, no transformers.

The fused step is not a different forward; it is the same forward with the
elementwise work moved inside the projection kernels. What that buys in
launches it costs in bookkeeping, and the bookkeeping is where a silent
wrong answer would come from:

* two sum-of-squares buffers alternating as the hand-off between a producing
  epilogue and the next RMSNorm prologue,
* a partial count per role that depends on the tile the tuner picked,
* a residual buffer that every RES epilogue updates in place.

So this runs the real ``Engine._decode_fused`` over a tiny random model with
``kernels.proj.project`` and the attention kernel swapped for PyTorch
equivalents, and compares its logits with ``Engine._decode_ref``. Anything
that survives on CPU can still be a slow kernel, but it is the right
arithmetic wired up the right way.

    python tests/test_fused_wiring.py
"""

import os
import sys
import types

import torch
import torch.nn.functional as F

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine")
sys.path.insert(0, ENGINE)

# kernels.proj imports triton at module scope; stub it, then never launch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_proj_plan import _stub_triton  # noqa: E402

_stub_triton()

import engine as engine_mod  # noqa: E402
import qwen  # noqa: E402
from kernels import proj, ref  # noqa: E402


# --------------------------------------------------------------------------
# A stand-in for the Transformers module tree qwen.Weights reads
# --------------------------------------------------------------------------

DIMS = dict(n_layers=3, nq=8, nkv=2, head_dim=32, hidden=256, inter=384, vocab=512)
ROPE_THETA = 5e6


def _w(*shape, scale=0.05):
    return torch.randn(*shape) * scale


class _Rotary:
    """Qwen3RotaryEmbedding, reduced to what ensure_rope calls."""

    def __init__(self, head_dim, theta):
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    def __call__(self, x, position_ids):
        freqs = position_ids.float()[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    d = DIMS
    h, i, v, nq, nkv, hd = (d["hidden"], d["inter"], d["vocab"],
                            d["nq"], d["nkv"], d["head_dim"])
    cfg = types.SimpleNamespace(
        num_hidden_layers=d["n_layers"], num_attention_heads=nq,
        num_key_value_heads=nkv, head_dim=hd, hidden_size=h,
        intermediate_size=i, rms_norm_eps=1e-6,
    )
    embed = _w(v, h, scale=0.3)
    layers = []
    for _ in range(d["n_layers"]):
        layers.append(types.SimpleNamespace(
            input_layernorm=types.SimpleNamespace(weight=1.0 + 0.3 * _w(h, scale=1.0)),
            post_attention_layernorm=types.SimpleNamespace(weight=1.0 + 0.3 * _w(h, scale=1.0)),
            self_attn=types.SimpleNamespace(
                q_norm=types.SimpleNamespace(weight=1.0 + 0.3 * _w(hd, scale=1.0)),
                k_norm=types.SimpleNamespace(weight=1.0 + 0.3 * _w(hd, scale=1.0)),
                q_proj=types.SimpleNamespace(weight=_w(nq * hd, h)),
                k_proj=types.SimpleNamespace(weight=_w(nkv * hd, h)),
                v_proj=types.SimpleNamespace(weight=_w(nkv * hd, h)),
                o_proj=types.SimpleNamespace(weight=_w(h, nq * hd)),
            ),
            mlp=types.SimpleNamespace(
                gate_proj=types.SimpleNamespace(weight=_w(i, h)),
                up_proj=types.SimpleNamespace(weight=_w(i, h)),
                down_proj=types.SimpleNamespace(weight=_w(h, i)),
            ),
        ))
    base = types.SimpleNamespace(
        embed_tokens=types.SimpleNamespace(weight=embed),
        norm=types.SimpleNamespace(weight=1.0 + 0.3 * _w(h, scale=1.0)),
        rotary_emb=_Rotary(hd, ROPE_THETA),
        layers=layers,
    )
    # Tied embedding and LM head, as the checkpoint has them.
    return types.SimpleNamespace(config=cfg, model=base,
                                 lm_head=types.SimpleNamespace(weight=embed))


# --------------------------------------------------------------------------
# PyTorch stand-ins with the launch signatures the fused step uses
# --------------------------------------------------------------------------


def project_shim(x, w, y, m, n, k, cfg, block_m, ssq_in, ssq_out, eps,
                 norm_w=None, n_parts=1, res=None, glu=False,
                 parts_block=proj.SSQ_PARTS, part=None):
    """``kernels.proj.project`` without Triton, including the partial layout.

    The partials are written per column tile, exactly as the epilogue writes
    one per program, so the consumer's ``n_parts`` is under test too.
    """
    if norm_w is None:
        assert n_parts == 1 or res is not None, "a non-normalising role read partials"
    else:
        assert n_parts <= parts_block, f"{n_parts} partials do not fit {parts_block}"
    xs = x[:m]
    if norm_w is not None:
        rstd = torch.rsqrt(ssq_in[:n_parts, :m].sum(0) / k + eps)
        xs = norm_w * (xs.to(torch.float32) * rstd[:, None]).to(xs.dtype)
    acc = F.linear(xs, w[:n] if glu else w)
    if glu:
        y[:m] = F.silu(acc) * F.linear(xs, w[n:])
    elif res is not None:
        h = (res[:m] + acc).to(res.dtype)
        res[:m] = h
        hf = h.to(torch.float32)
        for t in range(n // cfg.bn):
            ssq_out[t, :m] = hf[:, t * cfg.bn:(t + 1) * cfg.bn].pow(2).sum(-1)
    else:
        y[:m] = acc


def attn_shim(self, qkv, q_norm, k_norm, cos, sin, pos, k_cache, v_cache, eps, out=None):
    """``DecodeAttention.fused`` without Triton: head norm, RoPE and the cache
    write, then attention over ``[0, pos]``."""
    q = ref.qkv_post(qkv, q_norm, k_norm, cos, sin, pos, k_cache, v_cache,
                     1, self.nq, self.nkv, eps)
    o = ref.decode_attention(q, k_cache, v_cache, pos, self.nq, self.nkv)
    if out is None:
        return o
    out.copy_(o)
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def _build(batch, prompt_len, new_tokens):
    model = tiny_model()
    eng = engine_mod.Engine.from_model(model, "cpu", use_triton=False, use_graphs=False)
    st = engine_mod.ShapePlan(eng, batch, prompt_len, new_tokens)
    eng.w.ensure_rope(st.capacity)
    st.attn.fused = types.MethodType(attn_shim, st.attn)

    plan = engine_mod.FusedPlan(eng, batch)
    for role in plan.ROLES:
        n, k, _norm, _glu, _res = eng.w.role_dims(role)
        assert n % 16 == 0, f"{role}: n={n} is not a multiple of the test tile"
        plan.cfg[role] = proj.Config(16, 64)
    plan.note_producers()
    st.fused = plan
    return eng, st, plan


def test_fused_matches_the_reference_step():
    """Same prompt, same forced tokens: the fused step's logits must be the
    reference step's, up to FP32 summation order."""
    batch, prompt_len, steps = 3, 12, 4
    eng, st, _plan = _build(batch, prompt_len, steps)
    saved = engine_mod.kproj.project
    engine_mod.kproj.project = project_shim
    try:
        torch.manual_seed(11)
        prompt = torch.randint(0, DIMS["vocab"], (batch, prompt_len))
        toks = torch.randint(0, DIMS["vocab"], (steps, batch))

        eng._prefill(st, prompt)
        refs = []
        for t in range(steps):
            st.pos.fill_(prompt_len + t)
            st.ids.copy_(toks[t])
            refs.append(eng._decode_ref(st).clone())

        worst = 0.0
        for t in range(steps):
            st.pos.fill_(prompt_len + t)
            st.ids.copy_(toks[t])
            got = eng._decode_fused(st)
            assert torch.isfinite(got).all(), f"step {t}: non-finite logits"
            assert torch.equal(got.argmax(-1), refs[t].argmax(-1)), \
                f"step {t}: different token"
            worst = max(worst, (got - refs[t]).abs().max().item())
        scale = max(r.abs().max().item() for r in refs)
        assert worst <= 1e-4 * max(scale, 1.0), f"max|dlogit|={worst:.3g}"
    finally:
        engine_mod.kproj.project = saved


def test_partial_counts_match_the_tiles():
    """The consumer prologues read exactly what the producing epilogues
    wrote: O feeds gate/up, down feeds the next layer's QKV and the LM head."""
    eng, _st, plan = _build(2, 8, 2)
    hidden = eng.w.hidden
    assert plan.parts_o == hidden // plan.cfg["o"].bn
    assert plan.parts_down == hidden // plan.cfg["down"].bn
    assert plan.n_parts_in("gate_up") == plan.parts_o
    assert plan.n_parts_in("qkv") == plan.parts_down
    assert plan.n_parts_in("lm") == plan.parts_down
    assert plan.n_parts_in("o") == 1 and plan.n_parts_in("down") == 1
    assert plan.parts_block >= max(plan.parts_o, plan.parts_down)
    assert plan.parts_block & (plan.parts_block - 1) == 0, "PARTS must be a power of two"


def test_generate_contract_and_state_reset():
    """Exactly max_new_tokens steps, one id per sequence per step, and two
    calls in a row with different prompts must not share cache contents."""
    model = tiny_model()
    eng = engine_mod.Engine.from_model(model, "cpu", use_triton=False, use_graphs=False)
    torch.manual_seed(3)
    for batch, prompt_len, steps in ((1, 9, 5), (3, 16, 4), (2, 7, 1)):
        runs = []
        for _ in range(2):
            prompt = torch.randint(0, DIMS["vocab"], (batch, prompt_len)).tolist()
            got = list(eng.generate(prompt, steps))
            assert len(got) == steps, f"{len(got)} steps, wanted {steps}"
            assert all(isinstance(row, list) and len(row) == batch for row in got)
            assert all(all(isinstance(t, int) for t in row) for row in got)
            runs.append((prompt, got))
        # The same prompt fed again must reproduce its own continuation, which
        # it cannot do if the previous call left anything behind.
        again = list(eng.generate(runs[0][0], steps))
        assert again == runs[0][1], "state leaked between generate calls"


def test_residual_stream_is_updated_in_place():
    """Both RES epilogues write back into the one residual buffer, so after a
    step it holds the final-layer hidden state, not an intermediate."""
    batch, prompt_len = 2, 10
    eng, st, plan = _build(batch, prompt_len, 2)
    saved = engine_mod.kproj.project
    engine_mod.kproj.project = project_shim
    try:
        torch.manual_seed(5)
        prompt = torch.randint(0, DIMS["vocab"], (batch, prompt_len))
        eng._prefill(st, prompt)
        st.pos.fill_(prompt_len)
        st.ids.fill_(7)
        eng._decode_fused(st)
        h = plan.h
        assert torch.isfinite(h).all()
        # The LM head normalises h with the partials down left behind; if the
        # two disagreed the logits would be wrong by a scale factor.
        rstd = torch.rsqrt(plan.ssq_b[:plan.parts_down, :batch].sum(0) / eng.w.hidden
                           + eng.w.eps)
        want = torch.rsqrt(h.to(torch.float32).pow(2).mean(-1) + eng.w.eps)
        assert torch.allclose(rstd, want, rtol=1e-5, atol=1e-6), (rstd, want)
    finally:
        engine_mod.kproj.project = saved


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print("all good" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
