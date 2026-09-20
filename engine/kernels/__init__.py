"""Triton kernels for the Qwen3-4B decode engine.

``proj``  projection kernels; the whole fused decode step is built from them
``attn``  single-token GQA attention over the static KV cache
``misc``  elementwise kernels for prefill and the reference decode path
``ref``   PyTorch twins, used as the correctness arbiter and the fallback
"""
