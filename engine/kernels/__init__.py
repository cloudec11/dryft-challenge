"""Triton kernels for the decode step, and their PyTorch twins.

``gemm``         the projection kernel: every decode matrix product, with the
                 elementwise work fused in as a prologue or an epilogue, and
                 split-K reduced inside the same launch.
``attention``    single-token GQA over the fixed-capacity KV cache, with the
                 head norm, RoPE and cache write folded in.
``elementwise``  embedding, prefill head norm + RoPE, SwiGLU, RMSNorm.
``reference``    PyTorch twins.  Every kernel is checked against its twin at
                 load time and replaced by it if they disagree.
"""
