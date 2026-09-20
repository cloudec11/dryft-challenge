"""Qwen3-4B greedy decoder: hand-rolled forward, static KV cache, CUDA graphs.

What this engine is built around
--------------------------------

A decode step reads every weight exactly once -- 8.05 GB, of which 0.78 GB is
the tied LM head -- plus the KV cache.  On an H100 SXM that is a ~2.4 ms floor
at batch 16, and no rearrangement of the arithmetic changes it.  Measured
against that floor, the engines that came before this one ran at about 2.0
TB/s of *weights*, which reads as 60% of the device.  It was never 40% off
peak.  It was issuing 14.3 GB of loads to deliver 8.05 GB, because each of a
projection's ``N / BLOCK_N`` programs streams the whole input tile, and at
``BLOCK_N = 16`` that re-read is 6.25 GB per step.

So this engine attacks three things, in this order:

1. **Activation re-reads.**  Widen the column tile until the x term is small.
   That empties the device, so K is split to fill it back up -- the split adds
   no x traffic, because each slice program reads only its own part of the
   row.  The pair only works together, which is why neither v4 (split-K
   alone) nor v11 (wide tiles alone) moved the step.

2. **Launch count.**  ~185 short kernels per step, each paying wave fill and
   drain; a launch measured here costs ~2.8 us.  Every elementwise operation
   rides inside a projection as a prologue or an epilogue, the head norm,
   RoPE and cache write ride inside attention, and -- new here -- a split-K
   projection reduces its slices *inside the same launch*, so splitting is
   free of launches.  A layer is five kernels whatever the plan chooses.

3. **How the plan is chosen.**  ``planner`` scores the whole tile space on
   modelled traffic, with the bandwidth, launch cost and L2 price measured on
   the machine at warmup, and the tuner only races the handful the model
   cannot separate.  Racing a hand-written shortlist is what the previous
   engines did, and v11 spent its whole warmup budget doing it to arrive back
   where v8 was.

Correctness
-----------

Every Triton kernel has a PyTorch twin in ``kernels/reference.py``, is checked
against it at load time, and is replaced by the twin if it disagrees or fails
to compile.  The fused step is checked against a cuBLAS step's logits at the
real shape before it is allowed to run, and then has to win a timed race
against it.  BF16 rounding happens exactly where Transformers 4.51.3 rounds;
only the order of sums differs, which is what the 2.0-logit tie margin was
calibrated to allow.
"""

import contextlib
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

import model as checkpoint  # noqa: E402
import planner  # noqa: E402
from model import ROLES  # noqa: E402
from kernels import (  # noqa: E402
    attention as kattn,
    elementwise as kelem,
    gemm as kgemm,
    reference as kref,
)

try:  # PyTorch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
except ImportError:  # pragma: no cover
    SDPBackend = sdpa_kernel = None


# Decode steps queued ahead of the one being handed to the caller.  Only one
# is queued before the first token: a graph launch costs the host real time in
# this sandbox, and launches queued behind a short prefill would delay TTFT.
LOOKAHEAD = 3
# Prefills of at most this many tokens (batch x prompt) are tried as a CUDA
# graph and kept only if the replay beats the eager path.  Longer ones are
# GPU-bound anyway.
PREFILL_GRAPH_MAX_TOKENS = 16384
# Rows of a prefill processed at once in the row-wise stages.  Attention still
# sees whole sequences; this only bounds the MLP's activation footprint, which
# is 2.4 KB per row and would otherwise scale with batch times prompt.
PREFILL_CHUNK_ROWS = 16384
# Largest batch the fused step is tried at.  It is BLOCK_M of every
# projection, and tiles that need too much shared memory above it simply fail
# to compile during tuning and are skipped.
FUSED_MAX_BATCH = 256
# A later candidate must win by this much to displace an earlier one.  Runs
# vary 1-2% on identical code, so taking the bare minimum of a set of noisy
# measurements is how a tuner talks itself into a worse configuration; the
# candidate lists are ordered so that earlier means safer.  This guards the
# decisions taken *between* platform runs.
MARGIN = 0.97
# A tile race is a different kind of comparison: an in-process A/B on the same
# device in the same second, over 36 layers of real weights, minimum of
# rounds.  Its noise is well under 1%, so applying the between-run figure here
# would reject a 2% win on every role and keep the incumbent everywhere --
# which is exactly what v11 did.
TILE_MARGIN = 0.995
# Warmup seconds for the projection tiles and the attention tile.  The
# projection budget is spent once per distinct batch, not once per shape.
# Load plus one warmup generation share a 300-second budget, and going over
# it is ``timeout``: the whole run, not just the tuning.  So the two tuning
# budgets below are ceilings, not plans -- what each phase actually gets is
# whatever is left of WARMUP_BUDGET_S from the moment __init__ began, which is
# the only clock that matches the gate.  Loading the checkpoint is 30 to 60
# seconds of it and a Triton specialisation costs a few seconds to compile, so
# on a slow machine there may be nothing left, and the engine has to be as
# correct then as when there is plenty.
WARMUP_BUDGET_S = 255.0
# Held back from tuning for what has to happen afterwards whatever else does
# not: the fused step's check against the reference, the decode graph capture,
# and the warmup generation the platform is timing this budget against.
RESERVE_S = 35.0
# Below this much left, a plan is not even checked -- a check is a prefill
# and six eager steps, and its race is a graph capture.
FUSED_CHECK_S = 22.0
# How far below the reference step's argmax the fused step's own choice may
# sit before the plan is rejected.  The judge allows 2.0 and native Qwen
# drifts 0.75 against itself, so this is four times stricter than the rule it
# is protecting -- but it is not zero, because a genuine near-tie is not a
# wiring bug and rejecting one costs the whole workload the fused step.
TIE_OK = 0.5
# A plan raced against another plan, end to end, on the same device in the
# same second.  Tighter than MARGIN because there is no between-run noise in
# it, looser than TILE_MARGIN because being wrong here costs the whole step.
PLAN_MARGIN = 0.99
TUNE_BUDGET_S = 70.0
ATTN_TUNE_BUDGET_S = 25.0
CAND_LIMIT = 6
# Rows of the sum-of-squares hand-off buffer, as a power of two so the
# consumer's ``tl.arange`` covers it.
SSQ_PARTS = 256
# Cache slots beyond what a generation needs.  Warmup runs the step a few
# times past the prompt to compile, capture and race it, and attention writes
# slot ``pos`` every time; without slack those writes would run off the end of
# a cache sized exactly for the workload.
WARMUP_SLACK = 64


def _log(message):
    print(f"[engine] {message}", file=sys.stderr, flush=True)


class ShapePlan:
    """Buffers, tiles and graphs for one ``(batch, prompt, new tokens)``.

    Everything the decode step touches is allocated here and reused, so the
    captured graph's addresses never move and a step allocates nothing.
    """

    def __init__(self, engine, batch, prompt_len, new_tokens):
        w = engine.w
        device = engine.device
        self.key = (batch, prompt_len, new_tokens)
        self.batch = batch
        self.prompt_len = prompt_len
        self.new_tokens = new_tokens
        # The last decode step writes slot ``prompt + steps - 2``; the rest is
        # room for warmup to step past the end while it tunes.
        self.capacity = prompt_len + new_tokens + WARMUP_SLACK
        self.block_m = planner.block_m_for(batch)
        self.ss_rows = -(-batch // self.block_m) * self.block_m

        bf16 = w.dtype
        self.res = torch.zeros((batch, w.hidden), dtype=bf16, device=device)
        self.norm_buf = torch.zeros_like(self.res)
        self.proj_buf = torch.zeros_like(self.res)
        self.qkv = torch.zeros((batch, (w.nq + 2 * w.nkv) * w.head_dim),
                               dtype=bf16, device=device)
        self.attn_out = torch.zeros((batch, w.nq * w.head_dim), dtype=bf16,
                                    device=device)
        self.gate_up = torch.zeros((batch, 2 * w.inter), dtype=bf16, device=device)
        self.act = torch.zeros((batch, w.inter), dtype=bf16, device=device)
        self.logits = torch.zeros((batch, w.vocab), dtype=bf16, device=device)
        self.ids = torch.zeros(batch, dtype=torch.int64, device=device)
        self.pos = torch.zeros((), dtype=torch.int64, device=device)
        self.ssq_a = torch.ones((SSQ_PARTS, self.ss_rows), dtype=torch.float32,
                                device=device)
        self.ssq_b = torch.ones_like(self.ssq_a)

        shape = (w.n_layers, batch, w.nkv, self.capacity, w.head_dim)
        self.k_cache = torch.zeros(shape, dtype=bf16, device=device)
        self.v_cache = torch.zeros(shape, dtype=bf16, device=device)

        # Split-K scratch.  Sized for the widest role allowed to split, both
        # weight blocks of a GLU role counted, so one buffer serves them all.
        widest = max(2 * w.inter, self.qkv.shape[1], w.hidden)
        # One counter per column tile of the widest role that may split; the
        # LM head never splits, so its 9496 tiles do not have to be covered.
        max_tiles = (max(w.inter, self.qkv.shape[1], w.hidden) // 16
                     * (self.ss_rows // self.block_m) + 8)
        self.part, self.ctr = kgemm.buffers(
            self.ss_rows, widest, planner.MAX_SPLIT, max_tiles, device)
        self.partial_capacity = self.part.numel()

        self.tiles = {}
        self.attn = None
        self.graph = None
        self.prefill_graph = None
        self.step = None

        # Prefill working set.  The row-wise stages run in chunks so the MLP's
        # activations do not scale with batch times prompt.
        rows = batch * prompt_len
        # Whole sequences per chunk where that is possible, so the chunk
        # boundary never lands inside one.
        chunk = min(rows, max(prompt_len, PREFILL_CHUNK_ROWS - PREFILL_CHUNK_ROWS
                              % prompt_len))
        self.pre_rows = rows
        self.pre_chunk = chunk
        self.pre_ids = torch.zeros((batch, prompt_len), dtype=torch.int64,
                                   device=device)
        self.pre_x = torch.zeros((rows, w.hidden), dtype=bf16, device=device)
        self.pre_q = torch.zeros((rows, w.nq * w.head_dim), dtype=bf16, device=device)
        self.pre_attn = torch.zeros_like(self.pre_q)
        self.pre_norm = torch.zeros((chunk, w.hidden), dtype=bf16, device=device)
        self.pre_qkv = torch.zeros((chunk, self.qkv.shape[1]), dtype=bf16,
                                   device=device)
        self.pre_gate_up = torch.zeros((chunk, 2 * w.inter), dtype=bf16, device=device)
        self.pre_act = torch.zeros((chunk, w.inter), dtype=bf16, device=device)
        self.pre_proj = torch.zeros((chunk, w.hidden), dtype=bf16, device=device)
        self.pre_pos = torch.zeros((), dtype=torch.int64, device=device)
        self.tail = torch.zeros((batch, w.hidden), dtype=bf16, device=device)

        self.host = None
        self.events = None

    def n_parts(self, role, n):
        """Column tiles of a producing role, which is how many partial sums of
        squares the next RMSNorm reads."""
        tile = self.tiles.get(role)
        return 1 if tile is None else kgemm.n_tiles(tile, n)


class Engine:
    """The platform's entry point.  Two methods, and everything else private."""

    def __init__(self, model_path: str) -> None:
        # Before anything else, including the load: this is the clock the
        # 300-second gate runs on.
        self.started = start = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights = checkpoint.load(model_path, self.device)
        self._setup(weights, use_triton=True, use_graphs=True)
        _log(f"init {time.perf_counter() - start:.1f}s")

    @classmethod
    def from_model(cls, model, device, use_triton=True, use_graphs=True):
        """Build from an already-loaded Transformers model.  Offline tests."""
        self = cls.__new__(cls)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        self.device = torch.device(device)
        self._setup(checkpoint.Weights(model, self.device), use_triton, use_graphs)
        return self

    # ------------------------------------------------------------------ setup

    def _setup(self, weights, use_triton, use_graphs):
        if not hasattr(self, "started"):  # from_model, in the offline tests
            self.started = time.perf_counter()
        self.w = weights
        self.state = None
        self.use_graphs = use_graphs and self.device.type == "cuda"
        self.tile_cache = {}
        self.calibrated = False
        # One pool for every throwaway graph the tuner captures, taken from
        # the allocator rather than from a graph, so it stays valid after the
        # graph that first used it is dropped.
        self.pool = (torch.cuda.graph_pool_handle()
                     if self.device.type == "cuda" else None)

        sms = 132
        smem = planner.SMEM_PER_SM
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            sms = props.multi_processor_count
            smem = getattr(props, "shared_memory_per_multiprocessor", smem) or smem
            _log(f"device {props.name}, {sms} SMs, "
                 f"{props.total_memory / 2 ** 30:.0f} GiB, "
                 f"torch {torch.__version__}")
        self.dev = planner.Device(sms=sms, smem_per_sm=smem,
                                  smem_per_block=min(smem, planner.SMEM_PER_BLOCK))
        self._select_kernels(use_triton)

    def _select_kernels(self, use_triton):
        """Run every kernel against its twin and keep only the ones that agree.

        A kernel that fails to compile or disagrees is replaced by the twin for
        the whole run: the engine is slower than it could be, never wrong.
        """
        self.embed = kref.embed
        self.rms_norm = kref.add_rms_norm
        self.qkv_post = kref.qkv_post
        self.silu_mul = kref.silu_mul
        self.project = kref.project
        self.attn_triton = False
        self.fixup_ok = False
        self.gqa_sdpa = False
        self.argmax_out = False
        self.sdpa_ctx = contextlib.nullcontext
        live = []
        if not use_triton or self.device.type != "cuda":
            _log("kernels: reference (no CUDA device or Triton disabled)")
            return

        checks = (
            ("embed", self._check_embed, "embed", kelem.embed),
            ("rms_norm", self._check_rms_norm, "rms_norm", kelem.add_rms_norm),
            ("qkv_post", self._check_qkv_post, "qkv_post", kelem.qkv_post),
            ("silu_mul", self._check_silu_mul, "silu_mul", kelem.silu_mul),
            ("project", self._check_project, "project", kgemm.project),
        )
        for name, check, attr, impl in checks:
            try:
                ok = check()
            except Exception as exc:
                torch.cuda.synchronize()
                ok = False
                _log(f"kernel {name} unavailable: {type(exc).__name__}: {str(exc)[:200]}")
            if ok:
                setattr(self, attr, impl)
                live.append(name)
        for name, check, attr in (("attention", self._check_attention, "attn_triton"),
                                  ("fixup", self._check_fixup, "fixup_ok"),
                                  ("gqa_sdpa", self._check_gqa_sdpa, "gqa_sdpa"),
                                  ("flash_sdpa", self._check_flash_sdpa, "flash_sdpa"),
                                  ("argmax_out", self._check_argmax_out, "argmax_out")):
            try:
                ok = bool(check())
            except Exception as exc:
                torch.cuda.synchronize()
                ok = False
                _log(f"kernel {name} unavailable: {type(exc).__name__}: {str(exc)[:200]}")
            setattr(self, attr, ok)
            if ok:
                live.append(name)
        _log("kernels: " + (", ".join(live) if live else "none (reference only)"))

    # ------------------------------------------------------------ load checks

    def _randn(self, *shape, scale=1.0):
        return (torch.randn(shape, device=self.device, dtype=torch.float32)
                * scale).to(self.w.dtype)

    @staticmethod
    def _agree(fast, ref, tol=None):
        """Bit-identical is not the bar; the twin reorders sums too.  What is
        checked is that nothing structural differs: no NaN, no displaced row,
        no missing column."""
        f = fast.float()
        r = ref.float()
        if not torch.isfinite(f).all():
            return False
        scale = max(float(r.abs().max()), 1e-3)
        limit = tol if tol is not None else 0.02 * scale
        return float((f - r).abs().max()) <= limit

    def _check_embed(self):
        w = self.w
        ids = torch.randint(0, w.vocab, (4,), device=self.device)
        h1 = torch.zeros((4, w.hidden), dtype=w.dtype, device=self.device)
        h2 = torch.zeros_like(h1)
        s1 = torch.zeros((SSQ_PARTS, 4), dtype=torch.float32, device=self.device)
        s2 = torch.zeros_like(s1)
        kelem.embed(ids, w.embed_w, h1, s1)
        kref.embed(ids, w.embed_w, h2, s2)
        return self._agree(h1, h2, tol=0.0) and self._agree(s1[0], s2[0])

    def _check_rms_norm(self):
        w = self.w
        x = self._randn(5, w.hidden)
        r1 = self._randn(5, w.hidden)
        r2 = r1.clone()
        a = kelem.add_rms_norm(x, r1, w.norm_w, w.eps)
        b = kref.add_rms_norm(x, r2, w.norm_w, w.eps)
        return self._agree(a, b) and self._agree(r1, r2, tol=0.0)

    def _check_qkv_post(self):
        w = self.w
        w.ensure_rope(64)
        batch, T = 2, 3
        qkv = self._randn(batch * T, (w.nq + 2 * w.nkv) * w.head_dim)
        shape = (batch, w.nkv, 16, w.head_dim)
        k1 = torch.zeros(shape, dtype=w.dtype, device=self.device)
        v1 = torch.zeros_like(k1)
        k2 = torch.zeros_like(k1)
        v2 = torch.zeros_like(k1)
        pos = torch.zeros((), dtype=torch.int64, device=self.device)
        layer = w.layers[0]
        q1 = kelem.qkv_post(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, pos,
                            k1, v1, T, w.nq, w.nkv, w.eps)
        q2 = kref.qkv_post(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, pos,
                           k2, v2, T, w.nq, w.nkv, w.eps)
        return self._agree(q1, q2) and self._agree(k1, k2) and self._agree(v1, v2)

    def _check_silu_mul(self):
        gu = self._randn(3, 2 * self.w.inter)
        return self._agree(kelem.silu_mul(gu), kref.silu_mul(gu))

    def _check_project(self):
        """Every epilogue, both kernels, split and unsplit, against the twin.

        Small shapes on random data: what this is looking for is a wiring
        mistake -- an epilogue that writes the wrong buffer, a hand-off with
        the wrong stride, a GLU that pairs the wrong halves -- not a
        performance property.  The real shapes are covered later, by checking
        the whole fused step against the cuBLAS one.
        """
        w = self.w
        n, k = 512, 256
        ssq_in = torch.rand((SSQ_PARTS, 16), device=self.device) + 0.5
        for m in (5, 1):
            block_m = planner.block_m_for(m)
            x = self._randn(m, k, scale=0.1)
            weight = self._randn(2 * n, k, scale=0.05)
            norm_w = self._randn(k).float().abs().to(w.dtype) + 1.0
            part = torch.zeros(planner.MAX_SPLIT * block_m * 2 * n,
                               dtype=torch.float32, device=self.device)
            ctr = torch.zeros(n, dtype=torch.int32, device=self.device)
            tiles = [planner.Tile(block_m, 64, 64),
                     planner.Tile(block_m, 32, 128),
                     planner.Tile(block_m, 128, 64, 2, fixup=False)]
            if m == 1:
                tiles.append(planner.Tile(1, 64, 64, kind="vec"))
                tiles.append(planner.Tile(1, 128, 32, kind="vec"))
            for tile in tiles:
                for glu, res_mode, norm in ((False, False, True), (True, False, True),
                                            (False, True, False), (False, False, False)):
                    y1 = torch.zeros((m, n), dtype=w.dtype, device=self.device)
                    y2 = torch.zeros_like(y1)
                    r1 = self._randn(m, n) if res_mode else None
                    r2 = r1.clone() if res_mode else None
                    s1 = torch.zeros((SSQ_PARTS, 16), dtype=torch.float32,
                                     device=self.device)
                    s2 = torch.zeros_like(s1)
                    kgemm.project(x, weight, y1, m, n, k, tile, ssq_in, s1, w.eps,
                                  norm_w=norm_w if norm else None, glu=glu, res=r1,
                                  n_parts=3, ss_stride=16, part=part, ctr=ctr)
                    kref.project(x, weight, y2, m, n, k, tile, ssq_in, s2, w.eps,
                                 norm_w=norm_w if norm else None, glu=glu, res=r2,
                                 n_parts=3, ss_stride=16, part=part, ctr=ctr)
                    if not self._agree(*((r1, r2) if res_mode else (y1, y2))):
                        _log(f"project mismatch: m={m} {tile} glu={glu} "
                             f"res={res_mode} norm={norm}")
                        return False
                    if res_mode and not self._agree(s1[:n // tile.bn],
                                                    s2[:n // tile.bn]):
                        _log(f"project hand-off mismatch: m={m} {tile}")
                        return False
        return True

    def _check_fixup(self):
        """The in-kernel split-K reduction, which is the one piece of this
        engine that depends on cross-block memory ordering.

        Checked on a grid big enough that a broken publish would be seen --
        hundreds of programs, every one of them racing for the same counters
        -- and repeated, because a race that fails rarely still fails.
        """
        if self.project is not kgemm.project:
            return False
        w = self.w
        m, n, k = 16, 2048, 2048
        x = self._randn(m, k, scale=0.1)
        weight = self._randn(n, k, scale=0.05)
        part = torch.zeros(planner.MAX_SPLIT * 16 * n, dtype=torch.float32,
                           device=self.device)
        ctr = torch.zeros(n, dtype=torch.int32, device=self.device)
        ssq = torch.ones((SSQ_PARTS, 16), dtype=torch.float32, device=self.device)
        for split in (2, 4, 8):
            fixed = planner.Tile(16, 32, 64, split, fixup=True)
            plain = planner.Tile(16, 32, 64, split, fixup=False)
            for _ in range(6):
                y1 = torch.zeros((m, n), dtype=w.dtype, device=self.device)
                y2 = torch.zeros_like(y1)
                s1 = torch.zeros_like(ssq)
                s2 = torch.zeros_like(ssq)
                kgemm.project(x, weight, y1, m, n, k, fixed, ssq, s1, w.eps,
                              part=part, ctr=ctr, ss_stride=16)
                kgemm.project(x, weight, y2, m, n, k, plain, ssq, s2, w.eps,
                              part=part, ctr=ctr, ss_stride=16)
                if not self._agree(y1, y2, tol=0.0):
                    return False
                if int(ctr.abs().sum()) != 0:
                    return False  # a counter left dirty breaks the next replay
        return True

    def _check_attention(self):
        w = self.w
        w.ensure_rope(64)
        batch, capacity, pos = 3, 48, 20
        qkv = self._randn(batch, (w.nq + 2 * w.nkv) * w.head_dim)
        shape = (batch, w.nkv, capacity, w.head_dim)
        k1 = self._randn(*shape)
        v1 = self._randn(*shape)
        k2, v2 = k1.clone(), v1.clone()
        p = torch.full((), pos, dtype=torch.int64, device=self.device)
        layer = w.layers[0]
        ref = kref.decode_attention_fused(qkv, layer.q_norm, layer.k_norm, w.cos,
                                          w.sin, p, k2, v2, w.eps, w.nq, w.nkv)
        for splits, fixup in ((1, False), (4, False), (4, True), (7, True)):
            ka, va = k1.clone(), v1.clone()
            attn = kattn.DecodeAttention(batch, capacity, w.nq, w.nkv, w.head_dim,
                                         self.device, block_n=32, splits=splits,
                                         fixup=fixup)
            out = attn.fused(qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, p,
                             ka, va, w.eps)
            if not self._agree(out, ref) or not self._agree(ka, k2, tol=0.0):
                return False
            if fixup and int(attn.ctr.abs().sum()) != 0:
                return False
        return True

    def _check_gqa_sdpa(self):
        """``enable_gqa`` lets SDPA read the 8 KV heads directly instead of
        materialising 32 copies for every prefill token."""
        w = self.w
        q = self._randn(2, w.nq, 5, w.head_dim).view(2, w.nq, 5, w.head_dim)
        k = self._randn(2, w.nkv, 5, w.head_dim)
        v = self._randn(2, w.nkv, 5, w.head_dim)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        b = F.scaled_dot_product_attention(
            q, k.repeat_interleave(w.group, 1), v.repeat_interleave(w.group, 1),
            is_causal=True)
        return self._agree(a, b)

    def _check_flash_sdpa(self):
        """Pin the prefill to FlashAttention if it will take this shape.

        Left to the dispatcher, an SDPA call it cannot serve falls back to the
        math backend, which materialises the whole ``[B, 32, T, T]`` score
        matrix -- a gigabyte at batch 4 over 2048 tokens, and several times
        the time.  That is a large regression to discover from a run report,
        so it is settled here.

        The probe uses the layouts the prefill actually hands over: a
        transposed view of ``[B, T, nq, D]`` for the queries and a slice of
        the KV cache for the keys, because whether a backend accepts a call
        depends on those strides and not only on the head counts.
        """
        if sdpa_kernel is None or SDPBackend is None:
            return False
        w = self.w
        batch, length, capacity = 2, 40, 64
        q = self._randn(batch, length, w.nq, w.head_dim).transpose(1, 2)
        cache_k = self._randn(batch, w.nkv, capacity, w.head_dim)
        cache_v = self._randn(batch, w.nkv, capacity, w.head_dim)
        k, v = cache_k[:, :, :length], cache_v[:, :, :length]
        want = F.scaled_dot_product_attention(
            q, k.repeat_interleave(w.group, 1), v.repeat_interleave(w.group, 1),
            is_causal=True)
        if not self.gqa_sdpa:
            k = k.repeat_interleave(w.group, 1)
            v = v.repeat_interleave(w.group, 1)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            got = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 enable_gqa=self.gqa_sdpa)
        if not self._agree(got, want):
            return False
        self.sdpa_ctx = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION])
        return True

    def _check_argmax_out(self):
        x = torch.randn(3, 17, device=self.device)
        out = torch.zeros(3, dtype=torch.int64, device=self.device)
        torch.argmax(x, dim=-1, out=out)
        return bool((out == x.argmax(-1)).all())

    # ------------------------------------------------------------- timing kit

    def _graph_of(self, fn, before=None, share=True):
        """Capture ``fn`` after compiling and warming it.

        ``before`` is run ahead of every warmup call: a decode step advances
        the cache position, and warming it three times must not walk past the
        end of a cache sized for the workload.  ``share`` puts the graph in
        the pool the throwaway tuning graphs use; the graphs the run keeps get
        their own, because a pool is only safe to share between graphs whose
        captured allocations are dead by the time another one replays.
        """
        for _ in range(1):
            if before is not None:
                before()
            fn()
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                if before is not None:
                    before()
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if before is not None:
            before()
        graph = torch.cuda.CUDAGraph()
        if share and self.pool is not None:
            with torch.cuda.graph(graph, pool=self.pool):
                fn()
        else:
            with torch.cuda.graph(graph):
                fn()
        torch.cuda.synchronize()
        return graph

    @staticmethod
    def _time_replays(graph, rounds=3, reps=2, before=None):
        """Minimum over rounds of the mean of ``reps`` replays.

        The minimum is deliberate: a disturbed round is noise in one
        direction only, so the smallest observation is the best estimate of
        what the configuration costs when the machine is not busy elsewhere.
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = float("inf")
        for _ in range(rounds):
            if before is not None:
                before()
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / reps)
        return best

    def _race(self, timings, label, margin=MARGIN):
        """First entry wins unless a later one beats it by ``margin``."""
        best_key, best_time = timings[0]
        for key, seconds in timings[1:]:
            if seconds < best_time * margin:
                best_key, best_time = key, seconds
        detail = ", ".join(f"{k}:{t * 1e3:.3f}ms" for k, t in timings)
        _log(f"{label}: {detail} -> {best_key}")
        return best_key, best_time

    # ------------------------------------------------------------ calibration

    def _calibrate(self, st):
        """Measure the three constants the traffic model runs on.

        Bandwidth and the L2 price come from the same projection run at two
        column widths: the wide tile barely re-reads x, so it measures the
        stream on its own, and the difference between them is almost entirely
        the x term, which is what ``l2_cost`` prices.  The launch cost is the
        slope of two graphs that differ only in how many kernels they hold.
        """
        if self.calibrated or self.project is not kgemm.project:
            return
        self.calibrated = True
        w = self.w
        try:
            launch_s = self._measure_launch(st)
        except Exception as exc:
            torch.cuda.synchronize()
            launch_s = None
            _log(f"launch calibration skipped: {type(exc).__name__}: {str(exc)[:120]}")
        bw = l2 = None
        try:
            n, k, _, _, _ = w.role_dims("qkv")
            # Two tiles with the same grid and nearly the same occupancy, so
            # wave efficiency cancels and what is left between them is the
            # activation re-read: 31.5 MB against 3.9 MB for this role.  A
            # wide unsplit tile would have measured an empty device instead.
            narrow = planner.Tile(st.block_m, 16, 256)
            wide = planner.Tile(st.block_m, 128, 64, 8, fixup=self.fixup_ok)
            if k % (wide.split * wide.bk):
                wide = planner.Tile(st.block_m, 128, 64, 4, fixup=self.fixup_ok)
            tn = self._time_role(st, "qkv", narrow)
            tw = self._time_role(st, "qkv", wide)
            if tn and tw:
                bw, l2 = planner.fit_l2_cost((narrow, tn), (wide, tw),
                                             st.batch, n, k, False, self.dev)
        except Exception as exc:
            torch.cuda.synchronize()
            _log(f"bandwidth calibration skipped: {type(exc).__name__}: {str(exc)[:120]}")
        self.dev = self.dev.calibrated(bw=bw, launch_s=launch_s, l2_cost=l2)
        _log(f"calibrated {self.dev}")

    def _measure_launch(self, st):
        """Seconds a kernel launch costs inside a replayed graph.

        The slope of two graphs that differ only in how many kernels they
        hold, so whatever a replay costs to start is subtracted out.  The
        kernel itself is one row of 64 columns: as close to nothing as a
        launch can do.
        """
        if self.silu_mul is not kelem.silu_mul:
            return None
        src = torch.zeros((1, 128), dtype=self.w.dtype, device=self.device)
        dst = torch.zeros((1, 64), dtype=self.w.dtype, device=self.device)

        def body(count):
            def run():
                for _ in range(count):
                    self.silu_mul(src, out=dst)
            return run

        few = self._graph_of(body(32))
        many = self._graph_of(body(160))
        t_few = self._time_replays(few, rounds=3, reps=4) / 1e3
        t_many = self._time_replays(many, rounds=3, reps=4) / 1e3
        del few, many
        return (t_many - t_few) / 128.0

    # -------------------------------------------------------------- the plan

    def _parts_for(self, st, role):
        """Partial sums of squares this role's RMSNorm prologue will read.

        Known because the producing roles are tuned first: ``o`` hands to
        ``gate_up``, ``down`` hands to the next layer's ``qkv`` and to the LM
        head.  A role that does not normalise reads none.
        """
        w = self.w
        producer = {"qkv": "down", "gate_up": "o", "lm": "down"}.get(role)
        if producer is None:
            return 0
        return st.n_parts(producer, w.hidden)

    def _time_role(self, st, role, tile, rounds=3, reps=2):
        """Seconds for one launch of ``role`` with ``tile``, averaged over
        every layer's own weights.

        Sweeping all 36 layers is not thoroughness, it is the measurement: one
        layer's weights are 200 MB and a repeated single-layer benchmark would
        time an L2-resident read and choose a tile for a regime the real step
        never sees.  That is how v5 regressed.
        """
        w = self.w
        n, k, norm, glu, res = w.role_dims(role)
        weights = w.role_weights(role)
        x, y = self._role_buffers(st, role)
        norm_w = w.role_norm(role) if norm else None
        target = st.res if res else None
        parts = max(1, self._parts_for(st, role))

        def run():
            for weight in weights:
                self.project(x, weight, y, st.batch, n, k, tile, st.ssq_a, st.ssq_b,
                             w.eps, norm_w=norm_w, glu=glu, res=target,
                             n_parts=parts, ss_stride=st.ss_rows, part=st.part,
                             ctr=st.ctr)

        graph = self._graph_of(run)
        try:
            return self._time_replays(graph, rounds, reps) / 1e3 / len(weights)
        finally:
            del graph

    def _role_buffers(self, st, role):
        return {
            "qkv": (st.res, st.qkv),
            "o": (st.attn_out, st.proj_buf),
            "gate_up": (st.res, st.act),
            "down": (st.act, st.proj_buf),
            "lm": (st.res, st.logits),
        }[role]

    def _tune_projections(self, st, deadline):
        """Pick a tile for each role: model first, measurement last.

        The model orders the space; only the handful it cannot separate is
        compiled and raced.  A candidate has to beat the incumbent -- the tile
        that already worked at this batch -- by ``TILE_MARGIN`` to displace it.
        """
        w = self.w
        # Producers first: how many partials ``o`` and ``down`` leave behind
        # is part of what their consumers pay, so their tiles have to be
        # settled before the consumers are scored or timed.
        roles = ("o", "down", "qkv", "gate_up", "lm")
        share = max(2.0, (deadline - time.perf_counter()) / len(roles))
        for role in roles:
            n, k, _, glu, _ = w.role_dims(role)
            cached = self.tile_cache.get((st.batch, role))
            if cached is not None:
                st.tiles[role] = cached
                continue
            # The tile every engine from v1 to v12 ran, offered first so
            # the race starts from what is known to work and the model has to
            # earn the move.
            incumbent = planner.Tile(st.block_m, 16, 256)
            shortlist = planner.candidates(
                st.batch, n, k, glu, self.dev, limit=CAND_LIMIT,
                incumbent=incumbent if k % 256 == 0 else None,
                allow_split=role != "lm", allow_fixup=self.fixup_ok,
                max_parts=SSQ_PARTS if role in ("o", "down") else None,
                partial_capacity=st.partial_capacity,
                parts=self._parts_for(st, role),
            )
            if not shortlist:
                shortlist = [planner.Tile(st.block_m, 16, 256)]
            stop = min(deadline, time.perf_counter() + share)
            timings = []
            for tile in shortlist:
                if timings and time.perf_counter() > stop:
                    break
                try:
                    timings.append((tile, self._time_role(st, role, tile)))
                except Exception as exc:
                    torch.cuda.synchronize()
                    _log(f"tile {tile} for {role} skipped: "
                         f"{type(exc).__name__}: {str(exc)[:100]}")
            if not timings:
                st.tiles[role] = shortlist[0]
                continue
            best, seconds = self._race(timings, f"tile {role} B={st.batch}",
                                       margin=TILE_MARGIN)
            st.tiles[role] = best
            self.tile_cache[(st.batch, role)] = best
            bytes_moved = planner.traffic(best, st.batch, n, k, glu, self.dev)[0]
            _log(f"  {role}: {best} {bytes_moved / 1e6 / (seconds * 1e3):.0f} GB/s "
                 f"of weights, model {best.score * 1e6:.1f}us measured "
                 f"{seconds * 1e6:.1f}us")

    def _tune_attention(self, st, deadline):
        """Race the decode attention tile over every layer's own KV.

        Same rule as the projections: the benchmark has to touch as much
        distinct memory as the step does, so one call per layer over that
        layer's own cache, not one layer replayed.
        """
        w = self.w
        pos = min(st.capacity - 1, st.prompt_len + st.new_tokens // 2)
        st.pos.fill_(pos)
        best = None
        timings = []
        for block_n, splits, fixup in kattn.candidates(self.dev.sms, st.batch, w.nkv,
                                                       fixup=self.fixup_ok):
            if timings and time.perf_counter() > deadline:
                break
            try:
                attn = kattn.DecodeAttention(st.batch, st.capacity, w.nq, w.nkv,
                                             w.head_dim, self.device, block_n=block_n,
                                             splits=splits, fixup=fixup)
                if any(cand.splits == attn.splits and cand.block_n == attn.block_n
                       and cand.fixup == attn.fixup for cand, _ in timings):
                    continue
                layer_attn = attn

                def run():
                    for i in range(w.n_layers):
                        layer_attn.fused(st.qkv, w.layers[i].q_norm, w.layers[i].k_norm,
                                         w.cos, w.sin, st.pos, st.k_cache[i],
                                         st.v_cache[i], w.eps, out=st.attn_out)

                graph = self._graph_of(run)
                timings.append((attn, self._time_replays(graph, 3, 2) / 1e3 / w.n_layers))
                del graph
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"attn {block_n}x{splits} skipped: "
                     f"{type(exc).__name__}: {str(exc)[:100]}")
        if timings:
            best, _ = self._race(timings, f"attention B={st.batch} C={st.capacity}",
                                 margin=TILE_MARGIN)
        st.attn = best or kattn.DecodeAttention(st.batch, st.capacity, w.nq, w.nkv,
                                                w.head_dim, self.device)
        # Each candidate owns partial buffers sized by its split count; only
        # the winner's are wanted for the rest of the run.
        timings = None
        torch.cuda.empty_cache()
        st.pos.fill_(st.prompt_len)

    # -------------------------------------------------------------- the steps

    def _layer_range(self):
        return range(self.w.n_layers)

    def _step_safe(self, st):
        """A decode step out of cuBLAS and the checked leaf kernels.

        This is the floor: it needs no tile plan, no split-K and no fused
        epilogues, so whatever else fails, this runs.  It is also what the
        fused step is checked and raced against.
        """
        w = self.w
        torch.index_select(w.embed_w, 0, st.ids, out=st.res)
        for i in self._layer_range():
            layer = w.layers[i]
            self.rms_norm(st.res, None, layer.in_w, w.eps, out=st.norm_buf)
            torch.matmul(st.norm_buf, layer.qkv.t(), out=st.qkv)
            self._attend_decode(st, i, layer)
            torch.matmul(st.attn_out, layer.o.t(), out=st.proj_buf)
            st.res.add_(st.proj_buf)
            self.rms_norm(st.res, None, layer.post_w, w.eps, out=st.norm_buf)
            torch.matmul(st.norm_buf, layer.gate_up.t(), out=st.gate_up)
            self.silu_mul(st.gate_up, out=st.act)
            torch.matmul(st.act, layer.down.t(), out=st.proj_buf)
            st.res.add_(st.proj_buf)
        self.rms_norm(st.res, None, w.norm_w, w.eps, out=st.norm_buf)
        torch.matmul(st.norm_buf, w.lm_w.t(), out=st.logits)
        self._argmax_into(st.logits, st.ids)
        st.pos.add_(1)

    def _step_fast(self, st):
        """The planned step: five launches a layer, nothing between them.

        The sum-of-squares hand-off is what removes the norms: a residual
        epilogue leaves one partial per column tile in ``ssq_a``/``ssq_b``, and
        the next projection's prologue finishes the RMSNorm from them, so x is
        never read twice just to normalise it.  ``a`` carries the embedding and
        every layer's MLP output; ``b`` carries the attention output.
        """
        w = self.w
        self.embed(st.ids, w.embed_w, st.res, st.ssq_a)
        parts_in = 1
        parts_o = st.n_parts("o", w.hidden)
        parts_down = st.n_parts("down", w.hidden)
        nq, kq, _, _, _ = w.role_dims("qkv")
        no, ko, _, _, _ = w.role_dims("o")
        ng, kg, _, _, _ = w.role_dims("gate_up")
        nd, kd, _, _, _ = w.role_dims("down")
        nl, kl, _, _, _ = w.role_dims("lm")
        for i in self._layer_range():
            layer = w.layers[i]
            self.project(st.res, layer.qkv, st.qkv, st.batch, nq, kq,
                         st.tiles["qkv"], st.ssq_a, st.ssq_b, w.eps,
                         norm_w=layer.in_w, n_parts=parts_in, ss_stride=st.ss_rows,
                         part=st.part, ctr=st.ctr)
            self._attend_decode(st, i, layer)
            self.project(st.attn_out, layer.o, st.proj_buf, st.batch, no, ko,
                         st.tiles["o"], st.ssq_a, st.ssq_b, w.eps, res=st.res,
                         ss_stride=st.ss_rows, part=st.part, ctr=st.ctr)
            self.project(st.res, layer.gate_up, st.act, st.batch, ng, kg,
                         st.tiles["gate_up"], st.ssq_b, st.ssq_a, w.eps,
                         norm_w=layer.post_w, glu=True, n_parts=parts_o,
                         ss_stride=st.ss_rows, part=st.part, ctr=st.ctr)
            self.project(st.act, layer.down, st.proj_buf, st.batch, nd, kd,
                         st.tiles["down"], st.ssq_b, st.ssq_a, w.eps, res=st.res,
                         ss_stride=st.ss_rows, part=st.part, ctr=st.ctr)
            parts_in = parts_down
        self.project(st.res, w.lm_w, st.logits, st.batch, nl, kl, st.tiles["lm"],
                     st.ssq_a, st.ssq_b, w.eps, norm_w=w.norm_w, n_parts=parts_in,
                     ss_stride=st.ss_rows, part=st.part, ctr=st.ctr)
        self._argmax_into(st.logits, st.ids)
        st.pos.add_(1)

    def _attend_decode(self, st, i, layer):
        w = self.w
        if self.attn_triton and st.attn is not None:
            st.attn.fused(st.qkv, layer.q_norm, layer.k_norm, w.cos, w.sin, st.pos,
                          st.k_cache[i], st.v_cache[i], w.eps, out=st.attn_out)
        else:
            # The masked twin, not the indexed one: it reads ``pos`` on the
            # device, so the step stays capturable even with no attention
            # kernel at all.
            kref.decode_fused_masked(st.qkv, layer.q_norm, layer.k_norm, w.cos,
                                     w.sin, st.pos, st.k_cache[i], st.v_cache[i],
                                     w.eps, w.nq, w.nkv, out=st.attn_out)

    def _argmax_into(self, logits, ids):
        if self.argmax_out:
            torch.argmax(logits, dim=-1, out=ids)
        else:
            ids.copy_(torch.argmax(logits, dim=-1))

    # --------------------------------------------------------------- prefill

    def _prefill(self, st):
        """One full, unpadded pass over the prompt into an empty cache.

        Row-wise stages run in chunks; attention sees whole sequences.  The
        matrix products here are large enough that cuBLAS is the right kernel
        and the Triton tiles are not -- decode is where this engine's kernels
        earn their keep.
        """
        w = self.w
        batch, T = st.batch, st.prompt_len
        rows = st.pre_rows
        flat_ids = st.pre_ids.view(rows)
        torch.index_select(w.embed_w, 0, flat_ids, out=st.pre_x)
        st.pre_pos.zero_()
        chunk = st.pre_chunk
        for i in self._layer_range():
            layer = w.layers[i]
            for lo in range(0, rows, chunk):
                hi = min(lo + chunk, rows)
                c = hi - lo
                x = st.pre_x[lo:hi]
                nb = st.pre_norm[:c]
                self.rms_norm(x, None, layer.in_w, w.eps, out=nb)
                torch.matmul(nb, layer.qkv.t(), out=st.pre_qkv[:c])
                self.qkv_post(st.pre_qkv[:c], layer.q_norm, layer.k_norm, w.cos,
                              w.sin, st.pre_pos, st.k_cache[i], st.v_cache[i], T,
                              w.nq, w.nkv, w.eps, out=st.pre_q[lo:hi], row0=lo)
            self._attend_prefill(st, i)
            for lo in range(0, rows, chunk):
                hi = min(lo + chunk, rows)
                c = hi - lo
                x = st.pre_x[lo:hi]
                torch.matmul(st.pre_attn[lo:hi], layer.o.t(), out=st.pre_proj[:c])
                x.add_(st.pre_proj[:c])
                nb = st.pre_norm[:c]
                self.rms_norm(x, None, layer.post_w, w.eps, out=nb)
                torch.matmul(nb, layer.gate_up.t(), out=st.pre_gate_up[:c])
                self.silu_mul(st.pre_gate_up[:c], out=st.pre_act[:c])
                torch.matmul(st.pre_act[:c], layer.down.t(), out=st.pre_proj[:c])
                x.add_(st.pre_proj[:c])
        # The last row of each sequence, gathered into its own buffer: a
        # stride-T view is not something a kernel that indexes rows by
        # ``row * H`` can read.
        st.tail.copy_(st.pre_x.view(batch, T, w.hidden)[:, -1])
        self.rms_norm(st.tail, None, w.norm_w, w.eps, out=st.norm_buf)
        torch.matmul(st.norm_buf, w.lm_w.t(), out=st.logits)
        self._argmax_into(st.logits, st.ids)

    def _attend_prefill(self, st, i):
        """Causal GQA over the prompt.  ``enable_gqa`` keeps the eight KV heads
        as eight: expanding them to 32 is a copy of the whole cache slice."""
        w = self.w
        batch, T = st.batch, st.prompt_len
        q = st.pre_q.view(batch, T, w.nq, w.head_dim).transpose(1, 2)
        k = st.k_cache[i][:, :, :T]
        v = st.v_cache[i][:, :, :T]
        if not self.gqa_sdpa:
            k = k.repeat_interleave(w.group, 1)
            v = v.repeat_interleave(w.group, 1)
        try:
            with self.sdpa_ctx():
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                     enable_gqa=self.gqa_sdpa)
        except RuntimeError as exc:
            # The backend was pinned on a probe and refused the real call.
            # Give the dispatcher its choice back, once, for the whole run.
            _log(f"flash prefill refused this shape, unpinning: {str(exc)[:160]}")
            self.sdpa_ctx = contextlib.nullcontext
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 enable_gqa=self.gqa_sdpa)
        st.pre_attn.view(batch, T, w.nq, w.head_dim).copy_(out.transpose(1, 2))

    # ------------------------------------------------------- warmup and plan

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
        st.step = self._step_safe

        planning = self.project is kgemm.project and batch <= FUSED_MAX_BATCH
        if self.device.type == "cuda":
            # Everything from here is optional and is spent out of one budget.
            # The order is by what is lost if it is skipped: the tiles, then
            # the attention shape, then the calibration that only orders the
            # shortlist the tuner was going to measure anyway.
            spare = self._remaining() - RESERVE_S
            if spare > 0.4 * TUNE_BUDGET_S:
                self._calibrate(st)
            spare = self._remaining() - RESERVE_S
            if self.attn_triton and spare > 0:
                self._guard(self._tune_attention, "attention tuning", st,
                            time.perf_counter() + min(ATTN_TUNE_BUDGET_S,
                                                      0.3 * spare))
            spare = self._remaining() - RESERVE_S
            if planning and spare > 0:
                self._guard(self._tune_projections, "tile tuning", st,
                            time.perf_counter() + min(TUNE_BUDGET_S, spare))
        if planning and len(st.tiles) < len(ROLES):
            # Out of budget, or a role whose race raised.  The tile every
            # engine since v1 has run needs no measurement to be safe, and a
            # fused step on it still beats the cuBLAS one.
            self._fill_incumbent_tiles(st)
        if planning and st.tiles and self._remaining() > FUSED_CHECK_S:
            self._guard(self._adopt_fast_step, "fused step", st)
        if self.use_graphs and new_tokens > 1:
            # Never skipped for time: one capture is a few seconds and it is
            # the largest single win in the engine.
            self._guard(self._capture_decode, "decode graph", st)
        if (self.use_graphs and self.qkv_post is kelem.qkv_post
                and batch * prompt_len <= PREFILL_GRAPH_MAX_TOKENS
                and self._remaining() > 0.5 * RESERVE_S):
            # The reference qkv_post reads the position on the host, so a
            # prefill running on it cannot be captured at all.
            self._guard(self._capture_prefill, "prefill graph", st)

        self.state = st
        self._report(st, time.perf_counter() - start)
        return st

    def _remaining(self):
        """Seconds left of the load-plus-warmup budget."""
        return WARMUP_BUDGET_S - (time.perf_counter() - self.started)

    def _incumbent_plan(self, st):
        """The tile v1 through v12 all ran, for every role.

        It divides every one of this checkpoint's dimensions, it leaves 160
        partials for the two producing roles against a 256-row hand-off
        buffer, and it needs no split-K -- so it is legal without being
        checked, which is the point of having it.
        """
        plan = {}
        for role in ROLES:
            n, k, _, glu, _ = self.w.role_dims(role)
            incumbent = planner.Tile(st.block_m, 16, 256)
            if n % incumbent.bn == 0 and k % incumbent.bk == 0:
                plan[role] = incumbent
                continue
            # Not this checkpoint, then.  Take the model's first choice
            # unmeasured rather than launch a tile that does not divide.
            picks = planner.candidates(
                st.batch, n, k, glu, self.dev, limit=1,
                allow_split=role != "lm", allow_fixup=self.fixup_ok,
                max_parts=SSQ_PARTS if role in ("o", "down") else None,
                partial_capacity=st.partial_capacity)
            if picks:
                plan[role] = picks[0]
        return plan

    def _fill_incumbent_tiles(self, st):
        """Plan whatever the tuner did not get to, without measuring it."""
        fallback = self._incumbent_plan(st)
        for role in ROLES:
            if role in st.tiles:
                continue
            cached = self.tile_cache.get((st.batch, role))
            if cached is not None:
                st.tiles[role] = cached
            elif role in fallback:
                st.tiles[role] = fallback[role]
        _log(f"tiles: {len(st.tiles)} roles planned, "
             f"{self._remaining():.0f}s of budget left")

    def _guard(self, fn, label, *args):
        try:
            fn(*args)
            return True
        except Exception as exc:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                # An attempt that ran out of memory leaves its buffers behind;
                # the next one should not inherit the shortage.
                torch.cuda.empty_cache()
            _log(f"{label} unavailable: {type(exc).__name__}: {str(exc)[:300]}")
            return False

    def _adopt_fast_step(self, st):
        """Choose the decode step by measuring whole steps, not whole roles.

        A per-role race times one projection over 36 layers with nothing else
        running, which is not the state that projection meets inside a step:
        between two of its launches the step streams a layer's weights and its
        whole KV cache past the same L2. So the role races propose and this
        disposes -- the tuned plan has to beat the tile every engine since v1
        has run, end to end, on the real shape, or it does not ship.

        Both are checked for agreement first, and a plan that fails its check
        hands over to the next one rather than to cuBLAS: falling all the way
        back costs more than any tile choice can.
        """
        base = st.prompt_len

        def reset():
            st.pos.fill_(base)

        tuned = dict(st.tiles)
        incumbent = self._incumbent_plan(st)
        plans = []
        if (all(role in incumbent for role in ROLES)
                and any(incumbent[role].key != tuned[role].key for role in tuned)):
            plans.append(("incumbent", incumbent))  # first: it wins ties
        plans.append(("tuned", tuned))

        timings, kept = [], {}
        for name, tiles in plans:
            if timings and self._remaining() < FUSED_CHECK_S:
                break
            st.tiles = dict(tiles)
            if not self._guard(self._require_agreement, f"plan {name}", st):
                continue
            try:
                graph = self._graph_of(lambda: self._step_fast(st), before=reset)
            except Exception as exc:
                torch.cuda.synchronize()
                _log(f"plan {name} would not capture: "
                     f"{type(exc).__name__}: {str(exc)[:160]}")
                continue
            timings.append((name, self._time_replays(graph, before=reset) / 1e3))
            kept[name] = tiles
            del graph
        if not timings:
            st.tiles = tuned
            st.step = self._step_safe
            _log("no plan agreed with the reference step; decoding through cuBLAS")
            return

        choice, fast_s = self._race(timings, f"decode plan B={st.batch}",
                                    margin=PLAN_MARGIN)
        st.tiles = dict(kept[choice])
        safe = self._graph_of(lambda: self._step_safe(st), before=reset)
        safe_s = self._time_replays(safe, before=reset) / 1e3
        del safe
        winner, _ = self._race([("reference", safe_s), ("fused", fast_s)],
                               f"decode step B={st.batch}")
        st.step = self._step_fast if winner == "fused" else self._step_safe
        reset()

    def _require_agreement(self, st):
        """``_check_fast`` as something ``_guard`` can wrap: a plan that
        disagrees raises, so a plan that raises and a plan that disagrees take
        the same path."""
        if not self._check_fast(st):
            raise RuntimeError("logits disagree with the reference step")

    def _check_fast(self, st):
        """Run both steps from the same state and compare what they emit.

        The state is a real one.  An earlier version of this filled the KV
        cache with noise and compared logits against an absolute threshold,
        which is a worse test in both directions: the model is far out of
        distribution, so its logits are larger and two implementations that
        differ only in BF16 rounding drift further apart in absolute terms --
        and the run reports agree, because the engine scored what its cuBLAS
        step alone is worth.  A prefill of random token ids is both closer to
        what the judge sends and a better-conditioned comparison, and it costs
        a hundred milliseconds of a budget that has minutes in it.

        Three steps, not one: the hand-off buffers alternate between layers
        and between steps, so a wiring error that only appears on the second
        layer of the second step is exactly what this has to catch.

        The two runs share the cache rather than a copy of it.  Each writes
        slots ``base`` onward and reads ``[0, pos]``; the slots below ``base``
        are the prefill both runs start from and neither one touches.
        """
        w = self.w
        base = st.prompt_len
        st.pre_ids.copy_(torch.randint(0, w.vocab, (st.batch, st.prompt_len),
                                       device=self.device))
        self._prefill(st)
        seed_ids = st.ids.clone()

        def run(step):
            st.ids.copy_(seed_ids)
            st.pos.fill_(base)
            out = []
            for _ in range(3):
                step(st)
                out.append(st.logits.clone())
            return out

        ref = run(self._step_safe)
        got = run(self._step_fast)
        st.pos.fill_(base)
        worst = 0.0
        shortfall = 0.0
        for a, b in zip(got, ref):
            if not torch.isfinite(a).all():
                _log("plan rejected: non-finite logits")
                return False
            worst = max(worst, float((a.float() - b.float()).abs().max()))
            # What the judge would ask: the token this step would emit, scored
            # against the reference's own best. A near-tie flip is allowed and
            # does not cascade, because the replay follows the emitted prefix.
            chosen = b.float().gather(1, a.argmax(-1, keepdim=True))
            shortfall = max(shortfall, float((b.float().max(-1, keepdim=True).values
                                              - chosen).max()))
        ok = worst <= 1.0 and shortfall <= TIE_OK
        _log(f"plan vs reference step: max|dlogit| = {worst:.3f}, "
             f"token shortfall = {shortfall:.3f}, {'accepted' if ok else 'rejected'}")
        return ok

    def _capture_decode(self, st):
        base = st.prompt_len
        st.graph = self._graph_of(lambda: st.step(st),
                                  before=lambda: st.pos.fill_(base), share=False)
        st.pos.fill_(base)

    def _capture_prefill(self, st):
        """A prefill is ~430 launches at batch 1; a graph pays for them once.

        Kept only if the replay actually wins: on a long prefill the launches
        hide behind the matrix products and the graph is even.
        """
        st.pre_ids.copy_(torch.randint(0, self.w.vocab,
                                       (st.batch, st.prompt_len), device=self.device))
        graph = self._graph_of(lambda: self._prefill(st), share=False)
        eager = self._time_eager(lambda: self._prefill(st))
        replay = self._time_replays(graph, rounds=3, reps=1) / 1e3
        choice, _ = self._race([("eager", eager), ("graph", replay)],
                               f"prefill B={st.batch} S={st.prompt_len}")
        st.prefill_graph = graph if choice == "graph" else None
        if st.prefill_graph is None:
            del graph

    @staticmethod
    def _time_eager(fn, rounds=3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = float("inf")
        for _ in range(rounds):
            torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / 1e3)
        return best

    def _report(self, st, setup_s):
        """One line per shape, ending in the number that localises whatever
        gap is left: the fraction of HBM the step actually achieves."""
        w = self.w
        kv_bytes = (2 * st.batch * w.nkv * st.capacity * w.head_dim * 2 * w.n_layers)
        note = ""
        if st.graph is not None:
            try:
                ms = self._time_replays(st.graph, rounds=3, reps=2,
                                        before=lambda: st.pos.fill_(st.prompt_len))
                total = w.bytes_per_step + kv_bytes
                note = (f", step {ms:.3f} ms = {total / 1e6 / ms:.0f} GB/s over "
                        f"{total / 2 ** 30:.2f} GiB")
            except Exception:
                torch.cuda.synchronize()
            st.pos.fill_(st.prompt_len)
        tiles = " ".join(f"{role}={st.tiles[role]}" for role in sorted(st.tiles))
        peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                if self.device.type == "cuda" else 0.0)
        _log(f"shape B={st.batch} S={st.prompt_len} N={st.new_tokens}: "
             f"capacity {st.capacity}, {st.attn}, "
             f"step={'fused' if st.step is self._step_fast else 'reference'}, "
             f"graph={'yes' if st.graph else 'no'}, "
             f"prefill_graph={'yes' if st.prefill_graph else 'no'}, "
             f"peak {peak:.2f} GiB, setup {setup_s:.2f}s{note}")
        if tiles:
            _log(f"  tiles: {tiles}")

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
        exactly ``max_new_tokens`` times.  Every sequence in ``input_ids`` has
        the same length.  Does not stop at end-of-sequence tokens.
        """
        steps = int(max_new_tokens)
        if steps <= 0:
            return
        ids = torch.tensor(input_ids, dtype=torch.int64)
        batch, prompt_len = ids.shape
        # Planning and warmup stay outside inference mode: buffers allocated
        # inside it are inference tensors, and an inference tensor may not be
        # updated in place outside inference mode -- which is what a later
        # warmup, or a second generation, would do to them.
        st = self._state_for(batch, prompt_len, steps)
        if st.host is None or st.host.shape[0] < steps:
            st.host = torch.zeros((steps, batch), dtype=torch.int64,
                                  pin_memory=self.device.type == "cuda")
            st.events = ([torch.cuda.Event() for _ in range(steps)]
                         if self.device.type == "cuda" else None)
        with torch.inference_mode():
            # Everything prompt-dependent is rewritten here: cache slots
            # [0, S) by the prefill, then one slot per step, and attention
            # never reads past the current position.  Nothing survives from
            # the previous call.
            st.pre_ids.copy_(ids, non_blocking=True)
            if st.prefill_graph is not None:
                st.prefill_graph.replay()
            else:
                self._prefill(st)
            st.pos.fill_(prompt_len)
            self._publish(st, 0)

        # Keep the GPU ahead of the caller: each step's graph reads the token
        # the previous one wrote on the device, and its copy to the host is
        # queued right behind it.  Only one step is queued before the first
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
