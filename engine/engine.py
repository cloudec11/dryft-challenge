"""Qwen3-4B greedy decoder: hand-rolled forward, static KV cache, CUDA graphs.

Where the time goes, and what this engine does about it
-------------------------------------------------------

A decode step reads every weight exactly once -- 8.05 GB, of which 0.78 GB is
the tied LM head -- plus the KV cache. On an H100 SXM that is a ~2.4 ms floor
at batch 16 over 640 slots, and no rearrangement of the arithmetic changes
it. Measured against that floor this family of engines has run at about
2.0 TB/s of *weights*, which reads as 60% of the device -- but the step also
issues 6.25 GB of activation re-reads that never reach HBM, so it is moving
14.3 GB of loads in 4.8 ms and the memory system is closer to saturated than
idle. The engine is built around the three costs that follow from that:

1. **Launch count.** ~185 short kernels per step, each paying wave fill and
   drain. A launch measured here costs ~2.8 us, so the step's launches are
   ~0.5 ms on their own. Every elementwise operation rides inside a
   projection kernel as a prologue or an epilogue; the Q/K head norm, RoPE
   and the cache write ride inside attention; and attention skips its reduce
   kernel whenever one split covers the sequence. A layer is five launches.

2. **Activation re-reads.** Each of a projection's ``N/BLOCK_N`` programs
   streams the whole ``[BLOCK_M, K]`` input tile. At ``BLOCK_M = BLOCK_N =
   16`` -- the tile v1 through v11 all ran -- the x tile and the weight tile
   are exactly the same size, so half of every load is a re-read of the same
   80 KB of activations: 6.25 GB per step against 8.05 GB of weights. The
   step issues 14.3 GB of loads to deliver 8.05 GB, which is the whole
   reason it reads as 60% of HBM. ``kernels/proj.candidates`` scores tiles on
   that traffic, on wave quantisation (``N/BLOCK_N`` over 132 SMs, where a
   1.2-wave grid wastes 40% of the second wave) and on bytes in flight, and
   split-K is offered wherever a tile's own grid leaves the device
   underfilled -- which is what lets a wide tile pay for itself.

3. **Everything raced, nothing assumed.** Run-to-run timing noise on this
   platform is 1-2%, so a single measurement is weak evidence. Each race
   takes the minimum over rounds, starts from the configuration that already
   worked, and only moves off it for a win of ``MARGIN`` or better. Warmup
   benchmarks touch every layer's own weights and every layer's own KV, so
   they cannot measure an L2-resident read and then pick a tile for a regime
   the real step never sees.

Correctness
-----------

Every Triton kernel has a PyTorch twin in ``kernels/ref.py``, is checked
against it at load time, and is replaced by the twin if it disagrees or fails
to compile. The fused decode step is checked against the reference step's
logits at the real shape before it is allowed to run, and then has to win a
timed race against it. BF16 rounding happens exactly where Transformers
4.51.3 rounds; only the order of sums differs, which is what the 2.0-logit
tie margin was calibrated to allow.
"""

import os
import sys
import tempfile
import time


def _ensure_triton_cache():
    # The sandbox user may not own a writable home, and Triton needs a cache
    # directory before it compiles anything.
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

import qwen  # noqa: E402
from kernels import attn as kattn, misc as kmisc, proj as kproj, ref as kref  # noqa: E402

try:  # PyTorch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
except ImportError:  # pragma: no cover
    SDPBackend = sdpa_kernel = None


# Decode steps queued ahead of the one being handed to the caller. Only one
# is queued before the first token: a graph launch costs the host real time in
# this sandbox, and launches queued behind a short prefill would delay token 0.
LOOKAHEAD = 3
# Prefills of at most this many tokens (batch x prompt) are tried as a CUDA
# graph and kept only if the replay beats the eager path. Longer ones are
# GPU-bound anyway, and their activations would sit in a permanent pool.
PREFILL_GRAPH_MAX_TOKENS = 16384
# Largest batch the fused step is tried at; it is BLOCK_M of every projection,
# and tiles that need too much shared memory above it simply fail to compile
# during tuning and are skipped.
FUSED_MAX_BATCH = 128
# A later candidate must win by this much to displace an earlier one. Runs
# vary 1-2% on identical code, so taking the bare minimum of a set of noisy
# measurements is how a tuner talks itself into a worse configuration; the
# candidate lists are ordered so that earlier means safer.
MARGIN = 0.97
# The per-role tile race is a different kind of comparison and needs a
# different threshold. MARGIN guards decisions made *between* platform runs,
# where 1-2% is noise. A tile race is an in-process A/B on the same device in
# the same second, over 36 layers of real weights, minimum-of-rounds: its
# noise is well under 1%. Applying the between-run figure here rejects a 2%
# win on every one of five roles and keeps the incumbent everywhere -- which
# is exactly what v11 did, and why its B16 TPOT came back at 4.803 ms against
# v8's 4.808 despite searching a far wider space.
TILE_MARGIN = 0.995
# Warmup seconds for the projection tiles and for the attention tile, and how
# many candidates one projection may compile. The projection budget is spent
# once per distinct batch, not once per shape (see ``_tile_cache``).
TUNE_BUDGET_S = 80.0
ATTN_TUNE_BUDGET_S = 15.0
CAND_LIMIT = 10


def _log(message):
    print(f"[engine] {message}", file=sys.stderr, flush=True)


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class FusedPlan:
    """Tile choices and static buffers for the fused decode step at one batch.

    ``ROLES`` is in tuning order, not execution order: O and down produce the
    sums of squares that the three normalising roles consume, and how many
    partials there are depends on the tile they win with.
    """

    ROLES = ("o", "down", "qkv", "gate_up", "lm")

    def __init__(self, engine, batch):
        w = engine.w
        dev, dt = engine.device, engine.dtype
        layer = w.layers[0]
        self.m = batch
        self.bm = max(16, 1 << (batch - 1).bit_length())
        self.hidden = w.hidden
        self.inter = w.inter
        self.vocab = w.vocab
        self.n_qkv = layer.qkv.shape[0]
        self.k_o = layer.o.shape[1]

        self.h = torch.zeros((batch, self.hidden), dtype=dt, device=dev)
        self.qkv = torch.empty((batch, self.n_qkv), dtype=dt, device=dev)
        self.attn_out = torch.empty((batch, self.k_o), dtype=dt, device=dev)
        self.act = torch.empty((batch, self.inter), dtype=dt, device=dev)
        self.logits = torch.empty((batch, self.vocab), dtype=dt, device=dev)
        # Split-K is now offered to any role whose tile leaves the device
        # underfilled, so the partial buffer has to cover the widest of them.
        # The LM head never splits (1187 programs at its widest tile), and
        # gate/up cannot (two accumulators, one buffer), so this is the QKV
        # projection's 6144 columns.
        self.part = kproj.part_buffer(self.bm, max(self.hidden, self.n_qkv), dev)
        self.ssq_a = torch.zeros((kproj.SSQ_PARTS, self.bm), dtype=torch.float32, device=dev)
        self.ssq_b = torch.zeros_like(self.ssq_a)

        self.cfg = {}
        self.parts_o = 1
        self.parts_down = 1
        self.parts_block = 2

    def n_parts_in(self, role):
        """How many partials the prologue of ``role`` has to sum."""
        if role == "gate_up":
            return self.parts_o
        if role in ("qkv", "lm"):
            return self.parts_down
        return 1

    def note_producers(self):
        """Called once O and down have tiles; fixes the partial counts."""
        self.parts_o = kproj.n_tiles(self.cfg["o"], self.hidden)
        self.parts_down = kproj.n_tiles(self.cfg["down"], self.hidden)
        most = max(self.parts_o, self.parts_down)
        self.parts_block = max(2, 1 << (most - 1).bit_length())
        if self.parts_block > kproj.SSQ_PARTS:  # unreachable at hidden = 2560
            raise RuntimeError(f"{most} partials exceed the hand-off buffer")


class ShapePlan:
    """Buffers, tuned kernels and captured graphs for one workload shape."""

    def __init__(self, engine, batch, prompt_len, new_tokens):
        w = engine.w
        dev, dt = engine.device, engine.dtype
        self.key = (batch, prompt_len, new_tokens)
        self.batch = batch
        self.prompt_len = prompt_len
        self.new_tokens = new_tokens
        self.capacity = prompt_len + new_tokens
        shape = (w.n_layers, batch, w.nkv, self.capacity, w.head_dim)
        # Zeroed once, so a reference attention that masks over the whole
        # capacity never multiplies a zero probability by NaN garbage.
        self.k_cache = torch.zeros(shape, dtype=dt, device=dev)
        self.v_cache = torch.zeros(shape, dtype=dt, device=dev)
        self.ids = torch.zeros((batch,), dtype=torch.int64, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.attn = kattn.DecodeAttention(batch, self.capacity, w.nq, w.nkv,
                                          w.head_dim, dev)
        on_cuda = dev.type == "cuda"
        self.host = torch.empty((new_tokens, batch), dtype=torch.int64, pin_memory=on_cuda)
        self.events = [torch.cuda.Event() for _ in range(new_tokens)] if on_cuda else None
        self.graph = None
        self.step = engine._decode_ref
        self.fused = None
        self.prefill_graph = None
        self.prefill_ids = None
        self.prefill_out = None


class Engine:
    # ------------------------------------------------------------------ load

    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        start = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        _log(f"device {props.name}, {props.multi_processor_count} SMs, "
             f"{props.total_memory / 2 ** 30:.1f} GiB, torch {torch.__version__}")
        weights = qwen.load(model_path, device)
        self._setup(weights, use_triton=True, use_graphs=True)
        torch.cuda.empty_cache()
        _log(f"engine ready in {time.perf_counter() - start:.1f}s")

    @classmethod
    def from_model(cls, model, device, use_triton=True, use_graphs=True):
        """Build around an already-loaded Transformers model, for tests."""
        self = cls.__new__(cls)
        weights = qwen.Weights(model, torch.device(device))
        self._setup(weights, use_triton=use_triton, use_graphs=use_graphs)
        return self

    @torch.inference_mode()
    def _setup(self, weights, use_triton, use_graphs):
        self.w = weights
        self.device = weights.device
        self.dtype = weights.dtype
        self.use_graphs = use_graphs and self.device.type == "cuda"
        self.sm_count = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if self.device.type == "cuda" else 1
        )
        self.state = None
        self._seed = 1234
        # Tile choices depend on (batch, role) alone, never on prompt length
        # or output length, so a second shape at a batch already tuned reuses
        # the result instead of spending the budget again. With six hidden
        # workloads that is the difference between tuning once per batch and
        # once per workload.
        self._tile_cache = {}
        # Bytes of weight a decode step streams, for the throughput report.
        self.weight_bytes = sum(
            t.numel() * t.element_size()
            for layer in weights.layers
            for t in (layer.qkv, layer.o, layer.gate_up, layer.down)
        ) + weights.lm_w.numel() * weights.lm_w.element_size()
        self._select_kernels(use_triton)

    # -------------------------------------------------------- kernel choices

    def _select_kernels(self, use_triton):
        self.k_add_norm = kref.add_rms_norm
        self.k_qkv_post = kref.qkv_post
        self.k_silu_mul = kref.silu_mul
        self.k_embed = kref.embed
        self.use_triton_attn = False
        self.prefill_gqa = False
        self.fused_ok = use_triton and self.device.type == "cuda"
        # argmax straight into the id buffer saves a copy launch per step;
        # probe it rather than trusting the runtime's out= support.
        self.argmax_out = True
        try:
            probe = torch.empty((1,), dtype=torch.int64, device=self.device)
            torch.argmax(torch.zeros((1, 2), device=self.device), dim=-1, out=probe)
        except Exception:
            self.argmax_out = False

        chosen = {}
        if self.device.type == "cuda":
            try:
                ok, detail = self._check_prefill_gqa()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            self.prefill_gqa = ok
            chosen["prefill_attn"] = "flash-gqa" if ok else f"expanded ({detail})"
        if not use_triton:
            _log("kernels: PyTorch reference path; "
                 + ", ".join(f"{k}={v}" for k, v in chosen.items()))
            return

        for name, check in (
            ("add_norm", self._check_add_norm),
            ("qkv_post", self._check_qkv_post),
            ("silu_mul", self._check_silu_mul),
            ("embed", self._check_embed),
            ("decode_attn", self._check_decode_attn),
        ):
            try:
                ok, detail = check()
            except Exception as exc:  # compile or launch failure: keep PyTorch
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            chosen[name] = "triton" if ok else f"reference ({detail})"
            if not ok:
                continue
            if name == "add_norm":
                self.k_add_norm = kmisc.add_rms_norm
            elif name == "qkv_post":
                self.k_qkv_post = kmisc.qkv_post
            elif name == "silu_mul":
                self.k_silu_mul = kmisc.silu_mul
            elif name == "embed":
                self.k_embed = kmisc.embed
            elif name == "decode_attn":
                self.use_triton_attn = True
        _log("kernels: " + ", ".join(f"{k}={v}" for k, v in chosen.items()))

    # Load-time self-checks. A fused kernel must agree with its twin on nearly
    # every element; anything else is a bug, not a rounding difference.

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
        self._seed += 1
        g = torch.Generator(device="cpu").manual_seed(self._seed)
        return (torch.randn(*shape, generator=g) * scale).to(self.device, self.dtype)

    def _check_add_norm(self):
        hidden = self.w.hidden
        x, r = self._randn(5, hidden), self._randn(5, hidden, scale=4.0)
        weight = self.w.layers[0].post_w
        results = []
        for res in (None, r):
            a, ah = kmisc.add_rms_norm(x, res, weight, self.w.eps)
            b, bh = kref.add_rms_norm(x, res, weight, self.w.eps)
            results += [self._agree(a, b), self._agree(ah, bh)]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_qkv_post(self):
        w = self.w
        batch, T, cap = 2, 3, 64
        w.ensure_rope(cap)
        width = (w.nq + 2 * w.nkv) * w.head_dim
        qkv = self._randn(batch * T, width, scale=2.0)
        pos = torch.tensor([7], dtype=torch.int32, device=self.device)
        layer = w.layers[0]
        outs = []
        for fn in (kmisc.qkv_post, kref.qkv_post):
            kc = torch.zeros((batch, w.nkv, cap, w.head_dim), dtype=self.dtype,
                             device=self.device)
            vc = torch.zeros_like(kc)
            q = fn(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, pos, kc, vc,
                   T, w.nq, w.nkv, w.eps)
            outs.append((q, kc, vc))
        results = [self._agree(a, b) for a, b in zip(*outs)]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_silu_mul(self):
        gu = self._randn(3, 2 * self.w.inter, scale=3.0)
        return self._agree(kmisc.silu_mul(gu), kref.silu_mul(gu))

    def _check_embed(self):
        m = 5
        ids = torch.randint(0, self.w.vocab, (m,), device=self.device)
        outs = []
        for fn in (kmisc.embed, kref.embed):
            h = torch.zeros((m, self.w.hidden), dtype=self.dtype, device=self.device)
            ssq = torch.zeros((kproj.SSQ_PARTS, 16), dtype=torch.float32, device=self.device)
            fn(ids, self.w.embed_w, h, ssq)
            outs.append((h, ssq[0, :m]))
        results = [self._agree(a, b) for a, b in zip(*outs)]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_decode_attn(self):
        w = self.w
        batch, cap = 2, 300
        shape = (batch, w.nkv, cap, w.head_dim)
        kc, vc = self._randn(*shape, scale=2.0), self._randn(*shape)
        q = self._randn(batch, w.nq * w.head_dim, scale=2.0)
        results = []
        for p in (0, 70, cap - 1):
            pos = torch.tensor([p], dtype=torch.int32, device=self.device)
            attn = kattn.DecodeAttention(batch, cap, w.nq, w.nkv, w.head_dim, self.device)
            fast = attn.plain(q, kc, vc, pos)
            ref = kref.decode_attention(q, kc, vc, pos, w.nq, w.nkv)
            # A different but valid summation order: tolerance, not equality.
            results.append(self._agree(fast, ref, min_exact=0.0))
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)

    def _check_prefill_gqa(self):
        w = self.w
        if sdpa_kernel is None:
            return False, "no torch.nn.attention"
        batch, T, cap = 2, 80, 96
        q = self._randn(batch * T, w.nq * w.head_dim, scale=2.0)
        kc = self._randn(batch, w.nkv, cap, w.head_dim, scale=2.0)
        vc = self._randn(batch, w.nkv, cap, w.head_dim)
        fast = self._attend_prefill(q, kc, vc, batch, T, gqa=True)
        ref = self._attend_prefill(q, kc, vc, batch, T, gqa=False)
        return self._agree(fast, ref, min_exact=0.9)

    # --------------------------------------------------------------- forward

    def _layers_forward(self, x, T, pos, kc, vc, attend):
        """Reference forward. ``x`` is ``[B*T, H]``; returns the final-normed
        hidden state. Used by prefill, and by the decode step that the fused
        one is checked and raced against."""
        w = self.w
        n, res = self.k_add_norm(x, None, w.layers[0].in_w, w.eps)
        last = w.n_layers - 1
        for i, layer in enumerate(w.layers):
            qkv = F.linear(n, layer.qkv)
            q = self.k_qkv_post(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, pos,
                                kc[i], vc[i], T, w.nq, w.nkv, w.eps)
            o = F.linear(attend(q, kc[i], vc[i]), layer.o)
            m, res = self.k_add_norm(o, res, layer.post_w, w.eps)
            act = self.k_silu_mul(F.linear(m, layer.gate_up))
            d = F.linear(act, layer.down)
            next_w = w.norm_w if i == last else w.layers[i + 1].in_w
            n, res = self.k_add_norm(d, res, next_w, w.eps)
        return n

    def _attend_prefill(self, q, kc, vc, batch, T, gqa):
        """Causal attention of ``q`` over cache slots ``[0, T)``."""
        w = self.w
        q4 = q.view(batch, T, w.nq, w.head_dim).transpose(1, 2)
        if gqa:
            # FlashAttention with the 8 KV heads passed straight through: no
            # 32-head expansion copy, and no transpose copy of q.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                o = F.scaled_dot_product_attention(
                    q4, kc[:, :, :T], vc[:, :, :T], is_causal=True,
                    scale=w.scaling, enable_gqa=True,
                )
        else:
            group = w.nq // w.nkv
            k = _repeat_kv(kc[:, :, :T], group)
            v = _repeat_kv(vc[:, :, :T], group)
            o = F.scaled_dot_product_attention(q4.contiguous(), k, v, is_causal=True,
                                               scale=w.scaling)
        return o.transpose(1, 2).reshape(batch * T, w.nq * w.head_dim)

    def _prefill(self, st, ids):
        w = self.w
        batch, T = ids.shape
        zero = torch.zeros((1,), dtype=torch.int32, device=self.device)

        def attend(q, kc, vc):
            return self._attend_prefill(q, kc, vc, batch, T, self.prefill_gqa)

        x = F.embedding(ids.reshape(-1), w.embed_w)
        n = self._layers_forward(x, T, zero, st.k_cache, st.v_cache, attend)
        last = n.view(batch, T, -1)[:, -1, :]
        return torch.argmax(F.linear(last, w.lm_w), dim=-1)

    def _argmax_into(self, logits, ids):
        """Greedy pick, writing the id buffer in place. Ties go to the lowest
        index, as torch.argmax does."""
        if self.argmax_out:
            torch.argmax(logits, dim=-1, out=ids)
        else:
            ids.copy_(torch.argmax(logits, dim=-1))

    # ----------------------------------------------------------- decode step

    def _decode_ref(self, st):
        """One token per sequence at ``st.pos``, through the reference path.
        Graph-capturable; returns the step's logits."""
        w = self.w
        if self.use_triton_attn:
            def attend(q, kc, vc):
                return st.attn.plain(q, kc, vc, st.pos)
        else:
            def attend(q, kc, vc):
                return kref.decode_attention(q, kc, vc, st.pos, w.nq, w.nkv)

        x = F.embedding(st.ids, w.embed_w)
        n = self._layers_forward(x, 1, st.pos, st.k_cache, st.v_cache, attend)
        logits = F.linear(n, w.lm_w)
        self._argmax_into(logits, st.ids)
        st.pos.add_(1)
        return logits

    def _decode_fused(self, st):
        """The same step in five launches per layer.

        The residual stream lives in ``f.h`` and is updated in place by the O
        and down projections. ``ssq_a`` and ``ssq_b`` alternate as the
        hand-off for the sums of squares the next RMSNorm needs: ``a`` carries
        what O produced, ``b`` what down -- and, for the first layer, the
        embedding -- produced.
        """
        w, f = self.w, st.fused
        m, bm, cfg, eps = f.m, f.bm, f.cfg, w.eps
        pb, part = f.parts_block, f.part
        a, b = f.ssq_a, f.ssq_b

        self.k_embed(st.ids, w.embed_w, f.h, b)
        parts = 1
        for i, layer in enumerate(w.layers):
            kproj.project(f.h, layer.qkv, f.qkv, m, f.n_qkv, f.hidden, cfg["qkv"], bm,
                          b, a, eps, norm_w=layer.in_w, n_parts=parts,
                          parts_block=pb, part=part)
            att = st.attn.fused(f.qkv, layer.q_norm, layer.k_norm, w.cos, w.sin,
                                st.pos, st.k_cache[i], st.v_cache[i], eps, out=f.attn_out)
            kproj.project(att, layer.o, None, m, f.hidden, f.k_o, cfg["o"], bm,
                          b, a, eps, res=f.h, parts_block=pb, part=part)
            kproj.project(f.h, layer.gate_up, f.act, m, f.inter, f.hidden, cfg["gate_up"],
                          bm, a, b, eps, norm_w=layer.post_w, n_parts=f.parts_o,
                          glu=True, parts_block=pb, part=part)
            kproj.project(f.act, layer.down, None, m, f.hidden, f.inter, cfg["down"],
                          bm, a, b, eps, res=f.h, parts_block=pb, part=part)
            parts = f.parts_down
        kproj.project(f.h, w.lm_w, f.logits, m, f.vocab, f.hidden, cfg["lm"], bm,
                      b, a, eps, norm_w=w.norm_w, n_parts=parts,
                      parts_block=pb, part=part)
        self._argmax_into(f.logits, st.ids)
        st.pos.add_(1)
        return f.logits

    # ---------------------------------------------------------------- tuning

    @staticmethod
    def _time_replays(graph, rounds=4, reps=2):
        """Minimum over rounds of the mean replay time, in ms. The minimum
        rejects a round that lost the GPU to something else."""
        best = float("inf")
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        for _ in range(rounds):
            e0.record()
            for _ in range(reps):
                graph.replay()
            e1.record()
            e1.synchronize()
            best = min(best, e0.elapsed_time(e1) / reps)
        return best

    def _race(self, timings, label, margin=MARGIN):
        """Pick from ``[(ms, item), ...]``. The first entry is the incumbent,
        and a later one has to beat it by ``margin``."""
        best = timings[0]
        for entry in timings[1:]:
            if entry[0] < best[0] * margin:
                best = entry
        _log(f"  {label}: {best[1]} at {best[0] * 1000:.1f}us | "
             + " ".join(f"{e[1]}={e[0] * 1000:.1f}" for e in timings))
        return best

    def _tune_proj(self, plan, role, deadline):
        """Race tile shapes for one projection over all 36 layers' weights,
        inside a graph, so that nothing under test sits in L2."""
        w = self.w
        n, k, norm, glu, res = w.role_dims(role)
        m, bm = plan.m, plan.bm
        x = self._randn(m, k)
        out = torch.empty((m, n), dtype=self.dtype, device=self.device)
        res_buf = self._randn(m, n) if res else None
        # A plausible rstd: the live partials sum to about K, as they would
        # for a unit-variance row.
        n_parts = plan.n_parts_in(role)
        ssq_in = torch.full((kproj.SSQ_PARTS, bm), float(k) / max(n_parts, 1),
                            dtype=torch.float32, device=self.device)
        ssq_out = torch.zeros_like(ssq_in)
        norm_w = w.role_norm(role) if norm else None
        weights = w.role_weights(role)
        cands = kproj.candidates(m, n, k, glu, self.sm_count, limit=CAND_LIMIT,
                                 prefer_wide=res)

        timings = []
        for cfg in cands:
            if timings and time.perf_counter() > deadline:
                break

            def run(cfg=cfg):
                for weight in weights:
                    kproj.project(x, weight, None if res else out, m, n, k, cfg, bm,
                                  ssq_in, ssq_out, w.eps, norm_w=norm_w,
                                  n_parts=n_parts, res=res_buf, glu=glu,
                                  parts_block=plan.parts_block, part=plan.part)

            graph = None
            try:
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                timings.append((self._time_replays(graph) / len(weights), cfg))
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"  {role} {cfg}: skipped ({type(exc).__name__}: {str(exc)[:110]})")
            finally:
                del graph
        if not timings:
            raise RuntimeError(f"no tile ran for {role}")

        ms, best = self._race(timings, f"{role} n={n} k={k}", margin=TILE_MARGIN)
        moved = n * k * (2 if glu else 1) * 2  # bytes of weight per call
        _log(f"    {role}: {moved / 1e6 / ms:.0f} GB/s, {best.ctas} programs, "
             f"{best.waves:.2f} wave efficiency, {best.flight // 1024} KiB in flight/SM")
        return best

    def _tune_attention(self, st, deadline):
        """Race the decode attention's tile width and split count.

        One call per layer, over that layer's own cache. A single layer's KV
        fits in L2 at these shapes (42 MB at batch 16 over 640 slots), so
        replaying one call would measure an L2-resident read and choose a tile
        for a regime the real step never sees. In the step every layer's KV is
        cold.
        """
        w = self.w
        batch, cap = st.batch, st.capacity
        layer = w.layers[0]
        q = self._randn(batch, w.nq * w.head_dim, scale=2.0)
        qkv = self._randn(batch, layer.qkv.shape[0], scale=2.0)
        kc, vc = st.k_cache[0], st.v_cache[0]
        # Layer 0 gets real values so the correctness check below means
        # something; the rest stay zero, which costs the same bandwidth.
        kc.normal_(0.0, 2.0)
        vc.normal_(0.0, 1.0)
        pos = torch.tensor([cap - 1], dtype=torch.int32, device=self.device)
        use_fused = self.fused_ok and batch <= FUSED_MAX_BATCH
        out = torch.empty((batch, w.nq * w.head_dim), dtype=self.dtype, device=self.device)

        timings = []
        for cand in kattn.candidates(self.sm_count):
            block_n, target, warps, stages = cand
            graph = None
            try:
                attn = kattn.DecodeAttention(batch, cap, w.nq, w.nkv, w.head_dim,
                                             self.device, target_programs=target,
                                             block_n=block_n, num_warps=warps,
                                             num_stages=stages)

                def run(attn=attn):
                    for i in range(w.n_layers):
                        ki, vi = st.k_cache[i], st.v_cache[i]
                        if use_fused:
                            attn.fused(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin,
                                       pos, ki, vi, w.eps, out=out)
                        else:
                            attn.plain(q, ki, vi, pos, out=out)

                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                timings.append((self._time_replays(graph) / w.n_layers, attn))
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"  attn {cand}: skipped ({type(exc).__name__}: {str(exc)[:100]})")
            finally:
                del graph
            if timings and time.perf_counter() > deadline:
                break
        if not timings:
            return

        ms, attn = self._race(timings, "attn")
        ok, detail = self._agree(attn.plain(q, kc, vc, pos),
                                 kref.decode_attention(q, kc, vc, pos, w.nq, w.nkv),
                                 min_exact=0.0)
        kv_bytes = batch * w.nkv * cap * w.head_dim * 2 * 2
        _log(f"    attn: {kv_bytes / 1e6 / ms:.0f} GB/s of KV, {detail}")
        if ok:
            st.attn = attn

    def _check_fused(self, st, prompt_len, new_tokens):
        """The fused step's logits against the reference step's, on the same
        synthetic prompt and the same forced tokens, over a few steps."""
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
            refs.append(self._decode_ref(st).float().clone())

        max_d = mean_d = 0.0
        agree = 0
        for t in range(n_check):
            st.pos.fill_(prompt_len + t)
            st.ids.copy_(toks[t])
            got = self._decode_fused(st).float()
            if not torch.isfinite(got).all():
                return False, "non-finite logits"
            diff = (got - refs[t]).abs()
            max_d = max(max_d, diff.max().item())
            mean_d = max(mean_d, diff.mean().item())
            agree += (got.argmax(-1) == refs[t].argmax(-1)).sum().item()
        ok = max_d <= 1.5 and mean_d <= 0.1
        return ok, (f"max|dlogit|={max_d:.3f} mean={mean_d:.4f} "
                    f"argmax {agree}/{n_check * batch}")

    def _try_fused(self, st, prompt_len, new_tokens):
        start = time.perf_counter()
        plan = FusedPlan(self, st.batch)
        st.fused = plan
        # One slice of the budget per role, so early roles cannot starve the
        # last ones; time a role does not use carries forward.
        cached = self._tile_cache.get(st.batch)
        if cached is not None:
            plan.cfg.update(cached)
            plan.note_producers()
            _log(f"tiles for batch {st.batch} reused: "
                 + " ".join(f"{r}={plan.cfg[r]!r}" for r in plan.ROLES))
        else:
            slice_s = TUNE_BUDGET_S / len(plan.ROLES)
            for i, role in enumerate(plan.ROLES):
                plan.cfg[role] = self._tune_proj(plan, role, start + slice_s * (i + 1))
                if role == "down":
                    plan.note_producers()
            self._tile_cache[st.batch] = dict(plan.cfg)

        ok, detail = self._check_fused(st, prompt_len, new_tokens)
        _log(f"fused step vs reference: {detail} -> {'ok' if ok else 'REJECTED'}")
        if not ok:
            st.fused = None
            return

        graph = self._capture(st, self._decode_fused)
        reps = min(16, new_tokens)
        t_ref = min(self._time_step(st, st.graph, prompt_len, reps) for _ in range(2))
        t_fused = min(self._time_step(st, graph, prompt_len, reps) for _ in range(2))
        use = t_fused < t_ref * MARGIN
        _log(f"decode step: reference {t_ref:.3f} ms, fused {t_fused:.3f} ms -> "
             f"{'fused' if use else 'reference'} "
             f"(fused setup {time.perf_counter() - start:.1f}s)")
        if use:
            st.graph = graph
            st.step = self._decode_fused
        else:
            st.fused = None

    def _time_step(self, st, graph, prompt_len, reps):
        st.pos.fill_(prompt_len)
        graph.replay()
        st.pos.fill_(prompt_len)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            graph.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / reps

    # -------------------------------------------------------------- captures

    def _capture(self, st, step):
        # Warm on a side stream (this compiles kernels and creates cuBLAS
        # handles), then record. The warm steps scribble into cache slots that
        # the next prefill overwrites; no generation ever reads them.
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

    def _capture_prefill(self, st, batch, prompt_len):
        """Try replacing the prefill's ~430 launches with one replay.

        Only the prompt ids change between calls, so they live in a static
        buffer and the graph leaves the first token in ``st.prefill_out``. A
        short prefill is launch-bound and wins; a long one is GPU-bound, so
        the graph is dropped rather than holding its activations in a
        permanent pool. Wall clock, not events: part of the cost being
        measured is the host's.
        """
        start = time.perf_counter()
        st.prefill_ids = torch.zeros((batch, prompt_len), dtype=torch.int64,
                                     device=self.device)

        def timed(run):
            best = float("inf")
            for _ in range(2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t0) * 1e3)
            return best

        t_eager = timed(lambda: self._prefill(st, st.prefill_ids))
        torch.cuda.empty_cache()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._prefill(st, st.prefill_ids)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            st.prefill_out = self._prefill(st, st.prefill_ids)
        torch.cuda.synchronize()
        t_graph = timed(graph.replay)
        keep = t_graph < t_eager * MARGIN
        _log(f"prefill: eager {t_eager:.2f} ms, graph {t_graph:.2f} ms -> "
             f"{'graph' if keep else 'eager'} (setup {time.perf_counter() - start:.1f}s)")
        if keep:
            st.prefill_graph = graph
        else:
            st.prefill_out = None
            del graph
            torch.cuda.empty_cache()

    # ----------------------------------------------------------- shape setup

    def _state_for(self, batch, prompt_len, new_tokens):
        key = (batch, prompt_len, new_tokens)
        if self.state is not None and self.state.key == key:
            return self.state
        start = time.perf_counter()
        self.state = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        st = ShapePlan(self, batch, prompt_len, new_tokens)
        self.w.ensure_rope(st.capacity)

        if self.device.type == "cuda" and (self.use_triton_attn or self.fused_ok):
            try:
                self._tune_attention(st, time.perf_counter() + ATTN_TUNE_BUDGET_S)
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"attention tuning skipped: {type(exc).__name__}: {str(exc)[:200]}")
        if self.use_graphs and new_tokens > 1:
            try:
                st.graph = self._capture(st, self._decode_ref)
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
                    st.step = self._decode_ref
        if self.use_graphs and batch * prompt_len <= PREFILL_GRAPH_MAX_TOKENS:
            try:
                self._capture_prefill(st, batch, prompt_len)
            except Exception as exc:
                torch.cuda.synchronize()
                st.prefill_graph = None
                _log(f"prefill graph unavailable: {type(exc).__name__}: {str(exc)[:200]}")

        self.state = st
        self._report(st, time.perf_counter() - start)
        return st

    def _report(self, st, setup_s):
        """One line per shape, with the number that localises the remaining
        gap: what fraction of HBM the step actually achieves."""
        w = self.w
        kv_bytes = 2 * st.batch * w.nkv * st.capacity * w.head_dim * 2 * w.n_layers
        note = ""
        if st.graph is not None:
            try:
                ms = self._time_step(st, st.graph, st.prompt_len, min(16, st.new_tokens))
                st.pos.fill_(st.prompt_len)
                total = self.weight_bytes + kv_bytes
                note = (f", step {ms:.3f} ms = {total / 1e6 / ms:.0f} GB/s over "
                        f"{total / 2 ** 30:.2f} GiB")
            except Exception:
                torch.cuda.synchronize()
        _log(f"shape B={st.batch} S={st.prompt_len} N={st.new_tokens}: "
             f"capacity {st.capacity}, attn {st.attn} ({st.attn.splits} splits), "
             f"graph={'yes' if st.graph else 'no'}, "
             f"prefill_graph={'yes' if st.prefill_graph else 'no'}, "
             f"step={'fused' if st.fused is not None else 'reference'}, "
             f"setup {setup_s:.2f}s{note}")

    # ------------------------------------------------------------ generation

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
        exactly max_new_tokens times. Every sequence in input_ids has the
        same length. Does not stop at end-of-sequence tokens.
        """
        steps = int(max_new_tokens)
        if steps <= 0:
            return
        with torch.inference_mode():
            ids = torch.tensor(input_ids, dtype=torch.int64)
            batch, prompt_len = ids.shape
            st = self._state_for(batch, prompt_len, steps)
            # Everything prompt-dependent is rewritten here: cache slots
            # [0, S) by the prefill, then one slot per step, and attention
            # never reads past the current position. The fused step rebuilds
            # its residual and norm buffers from the token on every step.
            if st.prefill_graph is not None:
                st.prefill_ids.copy_(ids)
                st.prefill_graph.replay()
                st.ids.copy_(st.prefill_out)
            else:
                st.ids.copy_(self._prefill(st, ids.to(self.device, non_blocking=True)))
            st.pos.fill_(prompt_len)
            self._publish(st, 0)

        # Keep the GPU ahead of the caller: each step's graph reads the token
        # the previous one wrote on the device, and its copy to the host is
        # queued right behind it. Only one step is queued before the first
        # token, so launch time cannot push out TTFT.
        ahead = LOOKAHEAD if st.graph is not None else 1
        launched = 1
        for step in range(steps):
            target = min(steps, step + 1 + (1 if step == 0 else ahead))
            if launched < target:
                with torch.inference_mode():
                    while launched < target:
                        self._launch_step(st, launched)
                        launched += 1
            yield self._fetch(st, step)
