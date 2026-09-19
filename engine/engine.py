"""Qwen3 4B greedy decoder: hand-rolled forward, static KV cache, CUDA graphs.

What this changes relative to the Transformers baseline, and what it keeps:

* Weights are the loaded module's own tensors; Q/K/V and gate/up are packed
  into one matrix each so a layer issues fewer, larger GEMMs.
* Prefill runs eagerly and uses FlashAttention like the baseline, but passes
  the 8 KV heads directly (``enable_gqa``) instead of materialising 32
  expanded copies; it falls back to the baseline's expanded call if that
  path is unavailable or disagrees at load time.
* Decode is one CUDA graph per (batch, prompt, output) shape, captured during
  the untimed warmup call. It reads the token and position from device
  buffers, takes the argmax on the GPU and advances the position itself, so
  later steps are queued before earlier tokens are handed to the caller.
* Two decode steps exist. v1: cuBLAS projections plus fused Triton kernels
  (residual add + RMSNorm, head norm + RoPE + cache write, SwiGLU, split-K
  GQA attention). v2 (``kernels/fused.py``): skinny Triton GEMMs with the
  norms, SwiGLU and residual adds folded into them and the head norm + RoPE
  folded into attention, 6 launches per layer instead of 10. During warmup
  v2's tiles are tuned, its logits are checked against v1 on a synthetic
  prompt, both steps are captured and timed, and the faster one that agrees
  runs the samples.
* Every kernel rounds to BF16 at exactly the points the reference does; only
  reduction order differs. Each v1 kernel is checked against its PyTorch twin
  at load time and replaced by the twin if it disagrees or fails to compile.
"""

import os
import sys
import tempfile
import time


def _ensure_triton_cache():
    # The sandbox user may not own a writable home; Triton needs a cache dir.
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    home_cache = os.path.join(os.path.expanduser("~"), ".triton")
    try:
        os.makedirs(home_cache, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=home_cache, delete=True)
        probe.close()
    except OSError:
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-")


_ensure_triton_cache()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kernels import fused, ops  # noqa: E402

try:  # PyTorch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
except ImportError:  # pragma: no cover
    SDPBackend = sdpa_kernel = None

# Decode steps queued ahead of the one being handed to the caller.
LOOKAHEAD = 3
# Largest batch the fused decode step is tried at (it is BLOCK_M of its GEMMs).
FUSED_MAX_BATCH = 64
# Warmup seconds allowed for tuning the fused step's GEMM tiles.
TUNE_BUDGET_S = 90.0


def _log(message):
    print(f"[engine] {message}", file=sys.stderr, flush=True)


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class _Layer:
    __slots__ = ("in_w", "qkv", "q_norm", "k_norm", "o", "post_w", "gate_up", "down")


class _ShapeState:
    """Buffers and the captured graph for one (batch, prompt, output) shape."""

    def __init__(self, engine, batch, prompt_len, new_tokens):
        dev, dt = engine.device, engine.dtype
        self.key = (batch, prompt_len, new_tokens)
        self.batch = batch
        self.capacity = prompt_len + new_tokens
        shape = (engine.n_layers, batch, engine.nkv, self.capacity, engine.head_dim)
        # Zeroed once so the reference attention (which masks scores over the
        # full capacity) never multiplies a zero probability by NaN garbage.
        self.k_cache = torch.zeros(shape, dtype=dt, device=dev)
        self.v_cache = torch.zeros(shape, dtype=dt, device=dev)
        self.ids = torch.zeros((batch,), dtype=torch.int64, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.attn = ops.DecodeAttention(
            batch, self.capacity, engine.nq, engine.nkv, engine.head_dim, dev
        )
        on_cuda = dev.type == "cuda"
        self.host = torch.empty((new_tokens, batch), dtype=torch.int64, pin_memory=on_cuda)
        self.events = [torch.cuda.Event() for _ in range(new_tokens)] if on_cuda else None
        self.graph = None
        self.step = engine._decode_step  # eager step (fallback / capture source)
        self.fused = None  # _FusedPlan when the fused step is in use


class _FusedPlan:
    """Static buffers and tuned tile configs for the fused decode step."""

    ROLES = ("qkv", "o", "gate_up", "down", "lm")

    def __init__(self, engine, batch):
        dev, dt = engine.device, engine.dtype
        layer = engine.layers[0]
        self.m = batch
        self.bm = max(16, 1 << (batch - 1).bit_length())
        self.hidden = engine.norm_w.shape[0]
        self.inter = layer.down.shape[1]
        self.n_qkv = layer.qkv.shape[0]
        self.k_o = layer.o.shape[1]
        self.vocab = engine.lm_w.shape[0]
        self.h = torch.zeros((batch, self.hidden), dtype=dt, device=dev)
        self.qkv = torch.empty((batch, self.n_qkv), dtype=dt, device=dev)
        self.act = torch.empty((batch, self.inter), dtype=dt, device=dev)
        self.logits = torch.empty((batch, self.vocab), dtype=dt, device=dev)
        self.ssq_a = torch.zeros((fused.SSQ_PARTS, self.bm), dtype=torch.float32, device=dev)
        self.ssq_b = torch.zeros_like(self.ssq_a)
        self.cfg = {}
        self.parts_o = self.parts_down = 0

    def dims(self, role):
        """(n, k, norm, glu, res) of one projection role."""
        return {
            "qkv": (self.n_qkv, self.hidden, True, False, False),
            "o": (self.hidden, self.k_o, False, False, True),
            "gate_up": (self.inter, self.hidden, True, True, False),
            "down": (self.hidden, self.inter, False, False, True),
            "lm": (self.vocab, self.hidden, True, False, False),
        }[role]


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        from transformers import AutoModelForCausalLM

        start = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        _log(f"device {props.name}, {props.multi_processor_count} SMs, "
             f"{props.total_memory / 2**30:.1f} GiB, torch {torch.__version__}")
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        _log(f"checkpoint loaded in {time.perf_counter() - start:.1f}s")
        self._setup(model, torch.device("cuda:0"), use_triton=True, use_graphs=True)
        del model
        torch.cuda.empty_cache()
        _log(f"engine ready in {time.perf_counter() - start:.1f}s")

    @classmethod
    def from_model(cls, model, device, use_triton=True, use_graphs=True):
        """Build around an already-loaded Transformers model (for tests)."""
        self = cls.__new__(cls)
        self._setup(model, torch.device(device), use_triton=use_triton, use_graphs=use_graphs)
        return self

    # ------------------------------------------------------------------ setup

    @torch.inference_mode()
    def _setup(self, model, device, use_triton, use_graphs):
        cfg = model.config
        base = model.model
        self.device = device
        self.dtype = base.embed_tokens.weight.dtype
        self.n_layers = cfg.num_hidden_layers
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nq
        self.eps = cfg.rms_norm_eps
        self.scaling = self.head_dim ** -0.5
        self.use_graphs = use_graphs and device.type == "cuda"

        self.embed_w = base.embed_tokens.weight
        self.lm_w = model.lm_head.weight  # tied: the same tensor as embed_w
        self.norm_w = base.norm.weight
        self.rotary = base.rotary_emb
        self.layers = []
        for i in range(self.n_layers):
            src = base.layers[i]
            layer = _Layer()
            attn, mlp = src.self_attn, src.mlp
            layer.in_w = src.input_layernorm.weight
            layer.post_w = src.post_attention_layernorm.weight
            layer.q_norm = attn.q_norm.weight
            layer.k_norm = attn.k_norm.weight
            layer.qkv = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
            ).contiguous()
            layer.o = attn.o_proj.weight.contiguous()
            layer.gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
            layer.down = mlp.down_proj.weight.contiguous()
            # Drop the unpacked projections so they don't hold a second copy.
            base.layers[i] = torch.nn.Identity()
            self.layers.append(layer)
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self.cos = self.sin = None
        self.state = None
        self._select_kernels(use_triton)

    def _select_kernels(self, use_triton):
        self.k_add_norm = ops.add_rms_norm_ref
        self.k_qkv_post = ops.qkv_post_ref
        self.k_silu_mul = ops.silu_mul_ref
        self.use_triton_attn = False
        self.prefill_gqa = False
        self.fused_ok = use_triton and self.device.type == "cuda"
        chosen = {}
        if self.device.type == "cuda":
            try:
                ok, detail = self._check_prefill_gqa()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            self.prefill_gqa = ok
            chosen["prefill_attn"] = "flash-gqa" if ok else f"expanded ({detail})"
        if not use_triton:
            _log("kernels: PyTorch reference path; " + ", ".join(f"{k}={v}" for k, v in chosen.items()))
            return
        for name, check, fast in (
            ("add_norm", self._check_add_norm, ops.add_rms_norm),
            ("qkv_post", lambda: self._check_qkv_post(ops.qkv_post_rows), ops.qkv_post_rows),
            ("qkv_post_v1", lambda: self._check_qkv_post(ops.qkv_post), ops.qkv_post),
            ("silu_mul", self._check_silu_mul, ops.silu_mul),
            ("decode_attn", self._check_decode_attn, None),
        ):
            if name == "qkv_post_v1" and self.k_qkv_post is not ops.qkv_post_ref:
                continue  # the row kernel already passed
            try:
                ok, detail = check()
            except Exception as exc:  # compile or launch failure: keep PyTorch
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            chosen[name] = "triton" if ok else f"reference ({detail})"
            if ok and name == "add_norm":
                self.k_add_norm = fast
            elif ok and name in ("qkv_post", "qkv_post_v1"):
                self.k_qkv_post = fast
            elif ok and name == "silu_mul":
                self.k_silu_mul = fast
            elif ok and name == "decode_attn":
                self.use_triton_attn = True
        _log("kernels: " + ", ".join(f"{k}={v}" for k, v in chosen.items()))

    # Load-time self-checks. A fused kernel must agree with its PyTorch twin
    # bit-for-bit on nearly every element; anything else is a bug.

    @staticmethod
    def _agree(fast, ref, min_exact=0.98):
        fast, ref = fast.float(), ref.float()
        if not torch.isfinite(fast).all():
            return False, "non-finite"
        exact = (fast == ref).float().mean().item()
        err = (fast - ref).abs().max().item()
        scale = ref.abs().max().item()
        ok = exact >= min_exact and err <= 0.02 * scale + 1e-3
        return ok, f"exact={exact:.4f} max_err={err:.3g}"

    def _randn(self, *shape, scale=1.0):
        self._seed = getattr(self, "_seed", 1234) + 1
        g = torch.Generator(device="cpu").manual_seed(self._seed)
        return (torch.randn(*shape, generator=g) * scale).to(self.device, self.dtype)

    def _check_add_norm(self):
        hidden = self.norm_w.shape[0]
        x, r = self._randn(5, hidden), self._randn(5, hidden, scale=4.0)
        w = self.layers[0].post_w
        results = []
        for res in (None, r):
            a, ah = ops.add_rms_norm(x, res, w, self.eps)
            b, bh = ops.add_rms_norm_ref(x, res, w, self.eps)
            results += [self._agree(a, b), self._agree(ah, bh)]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_qkv_post(self, fast):
        batch, T, cap = 2, 3, 64
        self._ensure_rope(cap)
        width = (self.nq + 2 * self.nkv) * self.head_dim
        qkv = self._randn(batch * T, width, scale=2.0)
        pos = torch.tensor([7], dtype=torch.int32, device=self.device)
        layer = self.layers[0]
        outs = []
        for fn in (fast, ops.qkv_post_ref):
            kc = torch.zeros((batch, self.nkv, cap, self.head_dim), dtype=self.dtype, device=self.device)
            vc = torch.zeros_like(kc)
            q = fn(qkv, layer.q_norm, layer.k_norm, self.cos, self.sin, pos, kc, vc,
                   T, self.nq, self.nkv, self.eps)
            outs.append((q, kc, vc))
        results = [self._agree(a, b) for a, b in zip(*outs)]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_silu_mul(self):
        inter = self.layers[0].down.shape[1]
        gu = self._randn(3, 2 * inter, scale=3.0)
        return self._agree(ops.silu_mul(gu), ops.silu_mul_ref(gu))

    def _check_decode_attn(self):
        batch, cap = 2, 300
        shape = (batch, self.nkv, cap, self.head_dim)
        kc, vc = self._randn(*shape, scale=2.0), self._randn(*shape)
        q = self._randn(batch, self.nq * self.head_dim, scale=2.0)
        results = []
        for p in (0, 70, cap - 1):
            pos = torch.tensor([p], dtype=torch.int32, device=self.device)
            attn = ops.DecodeAttention(batch, cap, self.nq, self.nkv, self.head_dim, self.device)
            fast = attn(q, kc, vc, pos)
            ref = ops.decode_attention_ref(q, kc, vc, pos, self.nq, self.nkv)
            # Different (valid) summation order: compare with a tolerance only.
            results.append(self._agree(fast, ref, min_exact=0.0))
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_prefill_gqa(self):
        if sdpa_kernel is None:
            return False, "no torch.nn.attention"
        batch, T, cap = 2, 80, 96
        q = self._randn(batch * T, self.nq * self.head_dim, scale=2.0)
        kc = self._randn(batch, self.nkv, cap, self.head_dim, scale=2.0)
        vc = self._randn(batch, self.nkv, cap, self.head_dim)
        fast = self._attend_prefill(q, kc, vc, batch, T, gqa=True)
        ref = self._attend_prefill(q, kc, vc, batch, T, gqa=False)
        return self._agree(fast, ref, min_exact=0.9)

    # ---------------------------------------------------------------- forward

    def _ensure_rope(self, capacity):
        if self.cos is not None and self.cos.shape[0] >= capacity:
            return
        # Exactly the reference's tables: its own rotary module, BF16 output.
        positions = torch.arange(capacity, device=self.device)[None, :]
        probe = torch.empty((1,), dtype=self.dtype, device=self.device)
        cos, sin = self.rotary(probe, positions)
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()

    def _layers_forward(self, x, T, pos, kc, vc, attend):
        """``x`` is ``[B*T, H]``; returns the final-normed hidden ``[B*T, H]``."""
        n, res = self.k_add_norm(x, None, self.layers[0].in_w, self.eps)
        last = self.n_layers - 1
        for i, layer in enumerate(self.layers):
            qkv = F.linear(n, layer.qkv)
            q = self.k_qkv_post(qkv, layer.q_norm, layer.k_norm, self.cos, self.sin, pos,
                                kc[i], vc[i], T, self.nq, self.nkv, self.eps)
            o = F.linear(attend(q, kc[i], vc[i]), layer.o)
            m, res = self.k_add_norm(o, res, layer.post_w, self.eps)
            act = self.k_silu_mul(F.linear(m, layer.gate_up))
            d = F.linear(act, layer.down)
            next_w = self.norm_w if i == last else self.layers[i + 1].in_w
            n, res = self.k_add_norm(d, res, next_w, self.eps)
        return n

    def _attend_prefill(self, q, kc, vc, batch, T, gqa):
        """Causal attention of ``q`` ``[B*T, nq*D]`` over cache slots ``[0, T)``."""
        q4 = q.view(batch, T, self.nq, self.head_dim).transpose(1, 2)
        if gqa:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                o = F.scaled_dot_product_attention(
                    q4, kc[:, :, :T], vc[:, :, :T], is_causal=True, scale=self.scaling,
                    enable_gqa=True,
                )
        else:
            group = self.nq // self.nkv
            k = _repeat_kv(kc[:, :, :T], group)
            v = _repeat_kv(vc[:, :, :T], group)
            o = F.scaled_dot_product_attention(q4.contiguous(), k, v, is_causal=True, scale=self.scaling)
        return o.transpose(1, 2).reshape(batch * T, self.nq * self.head_dim)

    def _prefill(self, st, ids):
        batch, T = ids.shape
        zero = torch.zeros((1,), dtype=torch.int32, device=self.device)

        def attend(q, kc, vc):
            return self._attend_prefill(q, kc, vc, batch, T, self.prefill_gqa)

        x = F.embedding(ids.reshape(-1), self.embed_w)
        n = self._layers_forward(x, T, zero, st.k_cache, st.v_cache, attend)
        last = n.view(batch, T, -1)[:, -1, :]
        return torch.argmax(F.linear(last, self.lm_w), dim=-1)

    def _decode_step(self, st):
        """v1: one token per sequence at position ``st.pos``; graph-capturable.
        Returns the step's logits."""
        if self.use_triton_attn:
            def attend(q, kc, vc):
                return st.attn(q, kc, vc, st.pos)
        else:
            def attend(q, kc, vc):
                return ops.decode_attention_ref(q, kc, vc, st.pos, self.nq, self.nkv)

        x = F.embedding(st.ids, self.embed_w)
        n = self._layers_forward(x, 1, st.pos, st.k_cache, st.v_cache, attend)
        logits = F.linear(n, self.lm_w)
        st.ids.copy_(torch.argmax(logits, dim=-1))
        st.pos.add_(1)
        return logits

    def _decode_step_fused(self, st):
        """v2: the same step from ``kernels/fused.py``; graph-capturable.

        The residual stream lives in ``f.h`` and is updated in place by the O
        and down projections; ``ssq_a``/``ssq_b`` carry the per-row sums of
        squares that the next RMSNorm needs (a after O, b after down/embed)."""
        f = st.fused
        m, bm, c, eps, hid = f.m, f.bm, f.cfg, self.eps, f.hidden
        a_buf, b_buf = f.ssq_a, f.ssq_b
        fused.embed_ssq(st.ids, self.embed_w, f.h, b_buf)
        parts = 1
        for i, layer in enumerate(self.layers):
            fused.gemv(f.h, layer.qkv, f.qkv, m, f.n_qkv, hid, c["qkv"], bm, b_buf, a_buf, eps,
                       norm_w=layer.in_w, n_parts=parts)
            a = fused.attention_fused(st.attn, f.qkv, layer.q_norm, layer.k_norm, self.cos,
                                      self.sin, st.pos, st.k_cache[i], st.v_cache[i], eps)
            fused.gemv(a, layer.o, None, m, hid, f.k_o, c["o"], bm, b_buf, a_buf, eps, res=f.h)
            fused.gemv(f.h, layer.gate_up, f.act, m, f.inter, hid, c["gate_up"], bm, a_buf, b_buf,
                       eps, norm_w=layer.post_w, n_parts=f.parts_o, glu=True)
            fused.gemv(f.act, layer.down, None, m, hid, f.inter, c["down"], bm, a_buf, b_buf, eps,
                       res=f.h)
            parts = f.parts_down
        fused.gemv(f.h, self.lm_w, f.logits, m, f.vocab, hid, c["lm"], bm, b_buf, a_buf, eps,
                   norm_w=self.norm_w, n_parts=parts)
        st.ids.copy_(torch.argmax(f.logits, dim=-1))
        st.pos.add_(1)
        return f.logits

    # ------------------------------------------------------------ fused setup

    def _role_weights(self, role):
        if role == "lm":
            return [self.lm_w] * 4
        attr = {"qkv": "qkv", "o": "o", "gate_up": "gate_up", "down": "down"}[role]
        return [getattr(layer, attr) for layer in self.layers]

    def _role_norm(self, role):
        return {"qkv": self.layers[0].in_w, "gate_up": self.layers[0].post_w,
                "lm": self.norm_w}.get(role)

    def _tune_gemv(self, plan, role, deadline):
        """Time every candidate tile over all 36 layers' weights (so nothing
        sits in L2) inside a CUDA graph; return the fastest that runs."""
        n, k, norm, glu, res = plan.dims(role)
        m, bm = plan.m, plan.bm
        x = self._randn(m, k)
        out = torch.empty((m, n), dtype=self.dtype, device=self.device)
        res_buf = self._randn(m, n) if res else None
        ssq_in = torch.full((fused.SSQ_PARTS, bm), float(k), dtype=torch.float32, device=self.device)
        ssq_out = torch.zeros_like(ssq_in)
        norm_w = self._role_norm(role) if norm else None
        weights = self._role_weights(role)
        timings = []
        for cfg in fused.GEMV_CANDIDATES:
            if n % cfg.bn or k % cfg.bk:
                continue
            if timings and time.perf_counter() > deadline:
                break

            def run():
                for w in weights:
                    fused.gemv(x, w, None if res else out, m, n, k, cfg, bm, ssq_in, ssq_out,
                               self.eps, norm_w=norm_w, n_parts=1, res=res_buf, glu=glu)

            graph = None
            try:
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(3):
                    graph.replay()
                e1.record()
                e1.synchronize()
                timings.append((e0.elapsed_time(e1) * 1000.0 / (3 * len(weights)), cfg))
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"  {role} {cfg}: skipped ({type(exc).__name__}: {str(exc)[:120]})")
            finally:
                del graph
        if not timings:
            raise RuntimeError(f"no GEMV tile ran for {role}")
        timings.sort(key=lambda t: t[0])
        us, best = timings[0]
        moved = (n * k * (2 if glu else 1)) * 2 / 1e3  # KB of weight per call
        _log(f"  {role} n={n} k={k}: best {best} {us:.1f}us ({moved / us:.0f} GB/s); "
             + " ".join(f"{c}={t:.1f}" for t, c in timings[1:]))
        return best

    def _check_fused(self, st, prompt_len, new_tokens):
        """Logits of the fused step vs the v1 step on the same synthetic
        prompt and the same forced tokens, over a few consecutive steps."""
        batch = st.batch
        n_check = min(3, new_tokens)
        gen = torch.Generator(device="cpu").manual_seed(4321)
        prompt = torch.randint(100, 100000, (batch, prompt_len), generator=gen).to(self.device)
        toks = torch.randint(100, 100000, (n_check, batch), generator=gen).to(self.device)
        self._prefill(st, prompt)
        refs = []
        for t in range(n_check):
            st.pos.fill_(prompt_len + t)
            st.ids.copy_(toks[t])
            refs.append(self._decode_step(st).float())
        max_d = mean_d = 0.0
        agree = 0
        for t in range(n_check):
            st.pos.fill_(prompt_len + t)
            st.ids.copy_(toks[t])
            got = self._decode_step_fused(st).float()
            if not torch.isfinite(got).all():
                return False, "non-finite logits"
            diff = (got - refs[t]).abs()
            max_d = max(max_d, diff.max().item())
            mean_d = max(mean_d, diff.mean().item())
            agree += (got.argmax(-1) == refs[t].argmax(-1)).sum().item()
        ok = max_d <= 1.5 and mean_d <= 0.1
        return ok, f"max|dlogit|={max_d:.3f} mean={mean_d:.4f} argmax {agree}/{n_check * batch}"

    def _time_graph(self, st, graph, prompt_len, reps):
        st.pos.fill_(prompt_len)
        graph.replay()
        st.pos.fill_(prompt_len)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            graph.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / reps

    def _try_fused(self, st, prompt_len, new_tokens):
        start = time.perf_counter()
        plan = _FusedPlan(self, st.batch)
        st.fused = plan
        deadline = start + TUNE_BUDGET_S
        for role in plan.ROLES:
            plan.cfg[role] = self._tune_gemv(plan, role, deadline)
        plan.parts_o = plan.hidden // plan.cfg["o"].bn
        plan.parts_down = plan.hidden // plan.cfg["down"].bn
        ok, detail = self._check_fused(st, prompt_len, new_tokens)
        _log(f"fused step vs v1: {detail} -> {'ok' if ok else 'REJECTED'}")
        if not ok:
            st.fused = None
            return
        graph = self._capture(st, self._decode_step_fused)
        reps = min(16, new_tokens)
        t_v1, t_fused = [], []
        for _ in range(2):
            t_v1.append(self._time_graph(st, st.graph, prompt_len, reps))
            t_fused.append(self._time_graph(st, graph, prompt_len, reps))
        t_v1, t_fused = min(t_v1), min(t_fused)
        use = t_fused < t_v1
        _log(f"decode step: v1 {t_v1:.3f} ms, fused {t_fused:.3f} ms -> "
             f"{'fused' if use else 'v1'} (fused setup {time.perf_counter() - start:.1f}s)")
        if use:
            st.graph = graph
            st.step = self._decode_step_fused
        else:
            st.fused = None

    # ------------------------------------------------------------- generation

    def _state_for(self, batch, prompt_len, new_tokens):
        key = (batch, prompt_len, new_tokens)
        if self.state is not None and self.state.key == key:
            return self.state
        start = time.perf_counter()
        self.state = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        st = _ShapeState(self, batch, prompt_len, new_tokens)
        self._ensure_rope(st.capacity)
        if self.use_graphs and new_tokens > 1:
            try:
                st.graph = self._capture(st, self._decode_step)
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"graph capture failed, decoding eagerly: {type(exc).__name__}: {exc}")
                st.graph = None
            if st.graph is not None and self.fused_ok and batch <= FUSED_MAX_BATCH:
                try:
                    self._try_fused(st, prompt_len, new_tokens)
                except Exception as exc:
                    torch.cuda.synchronize()
                    _log(f"fused step unavailable: {type(exc).__name__}: {str(exc)[:300]}")
                    st.fused = None
                    st.step = self._decode_step
        self.state = st
        _log(f"shape B={batch} S={prompt_len} N={new_tokens}: capacity {st.capacity}, "
             f"{st.attn.splits} attention splits, graph={'yes' if st.graph else 'no'}, "
             f"step={'fused' if st.fused is not None else 'v1'}, "
             f"setup {time.perf_counter() - start:.2f}s")
        return st

    def _capture(self, st, step):
        # Warm on a side stream (compiles kernels, creates cuBLAS handles),
        # then record. The warm steps scribble into cache slots that the next
        # prefill overwrites; no generation ever reads them.
        st.pos.zero_()
        st.ids.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step(st)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(st)
        torch.cuda.synchronize()
        return graph

    def _launch_step(self, st, step):
        if st.graph is not None:
            st.graph.replay()
        else:
            st.step(st)
        self._publish(st, step)

    def _publish(self, st, step):
        st.host[step].copy_(st.ids, non_blocking=st.events is not None)
        if st.events is not None:
            st.events[step].record()

    def _fetch(self, st, step):
        if st.events is not None:
            st.events[step].synchronize()
        return st.host[step].tolist()

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        steps = int(max_new_tokens)
        if steps <= 0:
            return
        with torch.inference_mode():
            ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
            batch, prompt_len = ids.shape
            st = self._state_for(batch, prompt_len, steps)
            # Everything prompt-dependent is rewritten here: cache slots
            # [0, S) by prefill, then S onwards one step at a time; attention
            # never reads past the current position. The fused step's
            # residual and norm buffers are rebuilt from the token each step.
            st.ids.copy_(self._prefill(st, ids))
            st.pos.fill_(prompt_len)
            self._publish(st, 0)
        # Keep the GPU LOOKAHEAD steps ahead of the caller: each step's graph
        # reads the token the previous one wrote on the device, and its copy
        # to the host is queued right behind it.
        ahead = LOOKAHEAD if st.graph is not None else 1
        launched = 1
        for step in range(steps):
            target = min(steps, step + 1 + ahead)
            if launched < target:
                with torch.inference_mode():
                    while launched < target:
                        self._launch_step(st, launched)
                        launched += 1
            yield self._fetch(st, step)
