"""Qwen3 4B greedy decoder: hand-rolled forward, static KV cache, CUDA graphs.

What this changes relative to the Transformers baseline, and what it keeps:

* Weights are the loaded module's own tensors; Q/K/V and gate/up are packed
  into one matrix each so a layer issues fewer, larger GEMMs.
* Prefill runs eagerly and uses the same SDPA (FlashAttention) call as the
  baseline, on the same GQA-expanded keys and values.
* Decode is one CUDA graph per (batch, prompt, output) shape, captured during
  the untimed warmup call. It reads the token and position from device
  buffers, attends with a split-K Triton kernel over a fixed-capacity cache,
  takes the argmax on the GPU and advances the position itself, so step t+1
  is queued before step t's tokens are handed to the caller.
* Fused Triton kernels (residual add + RMSNorm, head norm + RoPE + cache
  write, SwiGLU) round to BF16 at exactly the points the reference does. Each
  one is checked against its PyTorch twin at load time and replaced by the
  twin if it disagrees or fails to compile.
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

from kernels import ops  # noqa: E402


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


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        from transformers import AutoModelForCausalLM

        start = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
            layer.o = attn.o_proj.weight
            layer.gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
            layer.down = mlp.down_proj.weight
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
        if not use_triton:
            _log("kernels: PyTorch reference path")
            return
        chosen = {}
        for name, check, fast in (
            ("add_norm", self._check_add_norm, ops.add_rms_norm),
            ("qkv_post", self._check_qkv_post, ops.qkv_post),
            ("silu_mul", self._check_silu_mul, ops.silu_mul),
            ("decode_attn", self._check_decode_attn, None),
        ):
            try:
                ok, detail = check()
            except Exception as exc:  # compile or launch failure: keep PyTorch
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            chosen[name] = "triton" if ok else f"reference ({detail})"
            if ok and name == "add_norm":
                self.k_add_norm = fast
            elif ok and name == "qkv_post":
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

    def _check_qkv_post(self):
        batch, T, cap = 2, 3, 64
        self._ensure_rope(cap)
        width = (self.nq + 2 * self.nkv) * self.head_dim
        qkv = self._randn(batch * T, width, scale=2.0)
        pos = torch.tensor([7], dtype=torch.int32, device=self.device)
        layer = self.layers[0]
        outs = []
        for fn in (ops.qkv_post, ops.qkv_post_ref):
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

    def _prefill(self, st, ids):
        batch, T = ids.shape
        zero = torch.zeros((1,), dtype=torch.int32, device=self.device)
        group = self.nq // self.nkv

        def attend(q, kc, vc):
            q4 = q.view(batch, T, self.nq, self.head_dim).transpose(1, 2).contiguous()
            k = _repeat_kv(kc[:, :, :T], group)
            v = _repeat_kv(vc[:, :, :T], group)
            o = F.scaled_dot_product_attention(q4, k, v, is_causal=True, scale=self.scaling)
            return o.transpose(1, 2).reshape(batch * T, self.nq * self.head_dim)

        x = F.embedding(ids.reshape(-1), self.embed_w)
        n = self._layers_forward(x, T, zero, st.k_cache, st.v_cache, attend)
        last = n.view(batch, T, -1)[:, -1, :]
        return torch.argmax(F.linear(last, self.lm_w), dim=-1)

    def _decode_step(self, st):
        """One token per sequence at position ``st.pos``; graph-capturable."""
        if self.use_triton_attn:
            def attend(q, kc, vc):
                return st.attn(q, kc, vc, st.pos)
        else:
            def attend(q, kc, vc):
                return ops.decode_attention_ref(q, kc, vc, st.pos, self.nq, self.nkv)

        x = F.embedding(st.ids, self.embed_w)
        n = self._layers_forward(x, 1, st.pos, st.k_cache, st.v_cache, attend)
        st.ids.copy_(torch.argmax(F.linear(n, self.lm_w), dim=-1))
        st.pos.add_(1)

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
                st.graph = self._capture(st)
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"graph capture failed, decoding eagerly: {type(exc).__name__}: {exc}")
                st.graph = None
        self.state = st
        _log(f"shape B={batch} S={prompt_len} N={new_tokens}: capacity {st.capacity}, "
             f"{st.attn.splits} attention splits, graph={'yes' if st.graph else 'no'}, "
             f"setup {time.perf_counter() - start:.2f}s")
        return st

    def _capture(self, st):
        # Warm on a side stream (compiles kernels, creates cuBLAS handles),
        # then record. The warm steps scribble into cache slots that the next
        # prefill overwrites; no generation ever reads them.
        st.pos.zero_()
        st.ids.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._decode_step(st)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._decode_step(st)
        torch.cuda.synchronize()
        return graph

    def _launch_step(self, st, step):
        if st.graph is not None:
            st.graph.replay()
        else:
            self._decode_step(st)
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
            # never reads past the current position.
            st.ids.copy_(self._prefill(st, ids))
            st.pos.fill_(prompt_len)
            self._publish(st, 0)
        for step in range(1, steps):
            # Queue step t before handing over step t-1: the GPU works while
            # the caller consumes tokens.
            with torch.inference_mode():
                self._launch_step(st, step)
            yield self._fetch(st, step - 1)
        yield self._fetch(st, steps - 1)
