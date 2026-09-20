"""Checkpoint loading, weight packing, and the facts the kernels compile to.

``__init__`` is untimed but shares a 300 s budget with one warmup call, so
everything that depends only on the checkpoint belongs here: read the
safetensors through Transformers (which keeps the tied embedding and LM head
a single tensor), concatenate the projections that are always used together,
and drop the Transformers modules so no second copy of a weight survives.

Packing buys fewer, larger reads per layer:

* ``qkv``      ``[nq*D + 2*nkv*D, H]``  = ``[6144, 2560]``
* ``gate_up``  ``[2*I, H]``             = ``[19456, 2560]``

Both stay ``[out, in]`` with the input contiguous, exactly as ``nn.Linear``
stores them, so ``linear(x) = x @ W.T`` and a kernel tile of ``BLOCK_N``
output columns reads ``BLOCK_N`` runs of contiguous input.

Weights are *not* relaid out beyond that.  A blocked ``[N/bn, K/bk, bn, bk]``
layout was considered and rejected: at ``BLOCK_K >= 64`` every row segment is
already 128 bytes of contiguous input, which is a whole L2 line and a full
DRAM burst, so the layout buys page locality at best -- and it would need a
second copy of all 8 GB, because prefill hands the same matrices to cuBLAS.
"""

import sys
import time

import torch

# Five roles, and what each one does with its input and output.  ``norm``
# means the kernel finishes an RMSNorm in its prologue from the partial sums
# of squares its producer left behind; ``glu`` that a second weight block
# streams beside the first; ``res`` that the result is added into the residual
# in place and the epilogue leaves the next RMSNorm's partials behind.
ROLES = ("qkv", "o", "gate_up", "down", "lm")


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
        # Kept alive only for its inv_freq buffer: building the tables with
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

    @property
    def bytes_per_step(self):
        """Weight bytes one decode step must read.  The floor everything else
        is measured against: 8.05 GB for this checkpoint."""
        layer = sum(w.numel() * w.element_size() for w in
                    (self.layers[0].qkv, self.layers[0].o,
                     self.layers[0].gate_up, self.layers[0].down))
        return layer * self.n_layers + self.lm_w.numel() * self.lm_w.element_size()

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
        """``(n, k, norm, glu, res)`` for one projection role."""
        layer = self.layers[0]
        return {
            "qkv": (layer.qkv.shape[0], self.hidden, True, False, False),
            "o": (self.hidden, layer.o.shape[1], False, False, True),
            "gate_up": (self.inter, self.hidden, True, True, False),
            "down": (self.hidden, self.inter, False, False, True),
            "lm": (self.vocab, self.hidden, True, False, False),
        }[role]

    def role_weights(self, role):
        """Every layer's weight for a role.

        A tuning sweep has to touch as much distinct memory as a real step
        does or it measures L2 and picks a tile for a regime the step never
        sees -- which is exactly how v5 regressed.
        """
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
         f"gate_up{list(weights.layers[0].gate_up.shape)} vocab={weights.vocab} "
         f"step={weights.bytes_per_step / 2 ** 30:.2f} GiB")
    return weights
