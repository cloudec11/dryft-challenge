"""Checkpoint loading, weight packing and the RoPE tables.

``__init__`` is untimed but shares a 300 s budget with one warmup call, so
everything that depends only on the checkpoint belongs here: read the
safetensors through Transformers (so the tied embedding / LM head stays one
tensor), concatenate the projections that are always used together, and drop
the Transformers modules so no second copy of a weight survives.

Packing buys fewer, larger reads per layer:

* ``qkv``      ``[nq*D + 2*nkv*D, H]``  = ``[6144, 2560]``
* ``gate_up``  ``[2*I, H]``             = ``[19456, 2560]``

Both are stored ``[out, in]`` with the input contiguous, exactly as
``nn.Linear`` stores them, so ``linear(x) = x @ W.T`` and a kernel tile of
``BLOCK_N`` output columns reads ``BLOCK_N`` runs of contiguous input.
"""

import sys
import time

import torch


def _log(message):
    print(f"[engine] {message}", file=sys.stderr, flush=True)


class LayerWeights:
    """One decoder layer, packed."""

    __slots__ = ("in_w", "post_w", "q_norm", "k_norm", "qkv", "o", "gate_up", "down")


class Weights:
    """Packed weights plus the dimensions the kernels need as constants."""

    def __init__(self, model, device):
        cfg = model.config
        base = model.model
        self.device = device
        self.dtype = base.embed_tokens.weight.dtype
        self.n_layers = cfg.num_hidden_layers
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nq
        self.group = self.nq // self.nkv
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.scaling = self.head_dim ** -0.5

        # Tied: lm_head.weight is embed_tokens.weight, so take both from the
        # loaded modules rather than re-reading the shards.
        self.embed_w = base.embed_tokens.weight
        self.lm_w = model.lm_head.weight
        self.norm_w = base.norm.weight
        self.vocab = self.lm_w.shape[0]
        # Kept alive only for its inv_freq buffer. Building the tables with
        # the reference's own module is what keeps RoPE from drifting.
        self.rotary = base.rotary_emb

        self.layers = []
        for i in range(self.n_layers):
            src = base.layers[i]
            attn, mlp = src.self_attn, src.mlp
            w = LayerWeights()
            w.in_w = src.input_layernorm.weight
            w.post_w = src.post_attention_layernorm.weight
            w.q_norm = attn.q_norm.weight
            w.k_norm = attn.k_norm.weight
            w.qkv = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
            ).contiguous()
            w.o = attn.o_proj.weight.contiguous()
            w.gate_up = torch.cat(
                [mlp.gate_proj.weight, mlp.up_proj.weight], dim=0
            ).contiguous()
            w.down = mlp.down_proj.weight.contiguous()
            # Release the unpacked projections as we go, so peak load memory
            # is one layer of duplication rather than a second whole model.
            base.layers[i] = torch.nn.Identity()
            self.layers.append(w)

        self.cos = None
        self.sin = None
        self._rope_capacity = 0

    # ------------------------------------------------------------------ rope

    def ensure_rope(self, capacity):
        """Build cos/sin for absolute positions ``[0, capacity)``.

        Uses the checkpoint's own rotary module, so the tables are bit for bit
        what Transformers would have produced, including the cast it applies
        on the way out.
        """
        if self.cos is not None and self._rope_capacity >= capacity:
            return
        positions = torch.arange(capacity, device=self.device)[None, :]
        probe = torch.empty((1,), dtype=self.dtype, device=self.device)
        cos, sin = self.rotary(probe, positions)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()
        self._rope_capacity = capacity

    # --------------------------------------------------------------- helpers

    def role_dims(self, role):
        """``(n, k, norm, glu, res)`` for one projection role.

        ``norm``  the kernel normalises its input in the prologue
        ``glu``   a second weight block follows the first (gate, then up)
        ``res``   the result is added into the residual buffer in place
        """
        layer = self.layers[0]
        return {
            "qkv": (layer.qkv.shape[0], self.hidden, True, False, False),
            "o": (self.hidden, layer.o.shape[1], False, False, True),
            "gate_up": (self.inter, self.hidden, True, True, False),
            "down": (self.hidden, self.inter, False, False, True),
            "lm": (self.vocab, self.hidden, True, False, False),
        }[role]

    def role_weights(self, role):
        """Every layer's weight for a role. A tuning sweep has to touch as
        much distinct memory as a real step does, or it measures L2."""
        if role == "lm":
            # One matrix, but 778 MB of it: four passes already evict L2.
            return [self.lm_w] * 4
        return [getattr(layer, role) for layer in self.layers]

    def role_norm(self, role):
        return {
            "qkv": self.layers[0].in_w,
            "gate_up": self.layers[0].post_w,
            "lm": self.norm_w,
        }.get(role)


def load(model_path, device):
    """Load the pinned checkpoint and return packed :class:`Weights`."""
    from transformers import AutoModelForCausalLM

    start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .to(device)
    )
    _log(f"checkpoint loaded in {time.perf_counter() - start:.1f}s")
    weights = Weights(model, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log(f"packed {weights.n_layers} layers: qkv{list(weights.layers[0].qkv.shape)} "
         f"gate_up{list(weights.layers[0].gate_up.shape)} vocab={weights.vocab}")
    return weights
