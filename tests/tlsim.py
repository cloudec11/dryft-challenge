"""Run a ``@triton.jit`` kernel's own source on numpy, one program at a time.

Triton 3.1's own interpreter (``TRITON_INTERPRET=1``) reads and writes through
raw device pointers and returns garbage for CPU tensors, so it cannot check a
kernel on a machine with no GPU.  This is the smallest thing that can: a
stand-in for ``triton.language`` covering exactly the operations the engine's
kernels use, over numpy arrays, with the grid walked in order.

What that buys, with no GPU anywhere:

* **Index arithmetic.**  Every masked load asserts that the elements it is
  actually reading are inside the buffer, so a stride or offset mistake fails
  here instead of silently reading a neighbouring tensor on the H100.
* **The epilogues and the hand-off.**  Whether the residual, the SwiGLU and
  the sum-of-squares land where the next kernel expects them.
* **The BF16 cast boundary.**  ``bfloat16`` is modelled as float32 rounded to
  8 mantissa bits after every cast, which is what BF16 arithmetic does, so
  ``(x * w).to(bf16)`` and ``x.to(bf16) * w`` are distinguishable here -- and
  the contract says one of them is out of budget.

What it does not buy: timing, occupancy, or anything about concurrency.  The
grid runs sequentially, so a kernel that races on a real device can pass here.
The split-K fixup's cross-block ordering is checked by the engine on the
device instead, against the two-launch path.
"""

import numpy as np

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


class DType:
    def __init__(self, name, np_dtype, bf16=False):
        self.name = name
        self.np = np_dtype
        self.bf16 = bf16
        self.scalar = self

    @property
    def element_ty(self):
        return self

    def __repr__(self):
        return f"tl.{self.name}"


float32 = DType("float32", np.float32)
float16 = DType("float16", np.float16)
bfloat16 = DType("bfloat16", np.float32, bf16=True)
int1 = DType("int1", np.bool_)
int32 = DType("int32", np.int32)
int64 = DType("int64", np.int64)
uint32 = DType("uint32", np.uint32)


def round_bf16(x):
    """Round float32 to BF16 precision, to nearest even, staying float32.

    This is what a BF16 multiply or add does on the device: the arithmetic
    happens in float32 and the result is rounded once on the way out.
    """
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    # round-to-nearest-even on bit 16
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    out = rounded.view(np.float32)
    return np.where(np.isnan(a), a, out)


def _cast(value, dtype):
    arr = np.asarray(value)
    if not isinstance(dtype, DType):  # a numpy dtype reached .to()
        return T(arr.astype(dtype))
    if dtype.bf16:
        return T(round_bf16(arr.astype(np.float32)))
    return T(arr.astype(dtype.np))


class T(np.ndarray):
    """A numpy array that also answers ``.to(dtype)``."""

    def __new__(cls, data):
        return np.asarray(data).view(cls)

    def to(self, dtype):
        return _cast(self, dtype)


def _t(x):
    return x if isinstance(x, T) else T(x)


# ---------------------------------------------------------------------------
# pointers
# ---------------------------------------------------------------------------


class Ptr:
    """A buffer plus an offset, which may be an array of offsets."""

    __slots__ = ("buf", "offset", "dtype", "name")

    def __init__(self, buf, dtype, offset=0, name="?"):
        self.buf = buf
        self.dtype = dtype
        self.offset = offset
        self.name = name

    def __add__(self, other):
        return Ptr(self.buf, self.dtype, self.offset + np.asarray(other),
                   self.name)

    __radd__ = __add__

    def __sub__(self, other):
        return Ptr(self.buf, self.dtype, self.offset - np.asarray(other), self.name)

    def indices(self):
        return np.asarray(self.offset, dtype=np.int64)


def pointer(array, dtype, name="?"):
    """Wrap a 1-D numpy buffer as a kernel argument."""
    return Ptr(array, DType(dtype.name, dtype.np, dtype.bf16), 0, name)


# ---------------------------------------------------------------------------
# the language
# ---------------------------------------------------------------------------

_STATE = {"pid": (0, 0, 0), "grid": (1, 1, 1)}


def set_program(pid, grid):
    _STATE["pid"] = pid
    _STATE["grid"] = grid


def program_id(axis):
    return T(np.int32(_STATE["pid"][axis]))


def num_programs(axis):
    return T(np.int32(_STATE["grid"][axis]))


def arange(lo, hi):
    return T(np.arange(lo, hi, dtype=np.int32))


def zeros(shape, dtype):
    return T(np.zeros(tuple(int(s) for s in shape), dtype=dtype.np))


def full(shape, value, dtype):
    return T(np.full(tuple(int(s) for s in shape), value, dtype=dtype.np))


def _broadcast(idx, mask, other):
    if mask is not None:
        idx, mask = np.broadcast_arrays(idx, np.asarray(mask))
    if other is not None and np.ndim(other) and mask is not None:
        other = np.broadcast_to(other, idx.shape)
    return idx, mask, other


def load(ptr, mask=None, other=0.0, cache_modifier=None, eviction_policy=None,
         volatile=False):
    idx, mask, other = _broadcast(ptr.indices(), mask, other)
    live = idx if mask is None else idx[mask]
    if live.size:
        lo, hi = int(live.min()), int(live.max())
        if lo < 0 or hi >= ptr.buf.size:
            raise IndexError(f"load from {ptr.name}[{lo}:{hi}] outside "
                             f"0:{ptr.buf.size}")
    out = np.zeros(idx.shape, dtype=ptr.buf.dtype)
    if mask is None:
        out = ptr.buf[idx]
    else:
        out = np.where(mask, ptr.buf[np.where(mask, idx, 0)], other)
    if ptr.dtype.bf16:
        return T(round_bf16(out.astype(np.float32)))
    return T(out)


def store(ptr, value, mask=None, cache_modifier=None, eviction_policy=None):
    idx = ptr.indices()
    value = np.asarray(value)
    if ptr.dtype.bf16:
        value = round_bf16(value.astype(np.float32))
    idx, mask, _ = _broadcast(idx, mask, None)
    value = np.broadcast_to(value, idx.shape)
    live = idx if mask is None else idx[mask]
    if live.size:
        lo, hi = int(live.min()), int(live.max())
        if lo < 0 or hi >= ptr.buf.size:
            raise IndexError(f"store to {ptr.name}[{lo}:{hi}] outside "
                             f"0:{ptr.buf.size}")
    if mask is None:
        ptr.buf[idx] = value.astype(ptr.buf.dtype)
    else:
        ptr.buf[idx[mask]] = value[mask].astype(ptr.buf.dtype)


def atomic_add(ptr, value, mask=None, sem=None, scope=None):
    """Scalar or elementwise atomic add; returns the old value(s)."""
    idx = ptr.indices()
    if np.ndim(idx) == 0:
        i = int(idx)
        old = ptr.buf[i]
        ptr.buf[i] = old + np.asarray(value)
        return T(old)
    idx = np.asarray(idx)
    old = ptr.buf[idx].copy()
    add = np.broadcast_to(np.asarray(value), idx.shape)
    if mask is None:
        ptr.buf[idx] = old + add
    else:
        m = np.asarray(mask)
        ptr.buf[idx[m]] = old[m] + add[m]
    return T(old)


def atomic_xchg(ptr, value, mask=None, sem=None, scope=None):
    idx = ptr.indices()
    if np.ndim(idx) == 0:
        i = int(idx)
        old = ptr.buf[i]
        ptr.buf[i] = np.asarray(value)
        return T(old)
    idx = np.asarray(idx)
    old = ptr.buf[idx].copy()
    new = np.broadcast_to(np.asarray(value), idx.shape)
    if mask is None:
        ptr.buf[idx] = new
    else:
        m = np.asarray(mask)
        ptr.buf[idx[m]] = new[m]
    return T(old)


def debug_barrier():
    return None


def where(condition, a, b):
    return T(np.where(np.asarray(condition), np.asarray(a), np.asarray(b)))


def sum(x, axis=None):
    return T(np.sum(np.asarray(x, dtype=np.float32), axis=axis))


def max(x, axis=None):
    return T(np.max(np.asarray(x), axis=axis))


def min(x, axis=None):
    return T(np.min(np.asarray(x), axis=axis))


def maximum(a, b):
    return T(np.maximum(np.asarray(a), np.asarray(b)))


def minimum(a, b):
    return T(np.minimum(np.asarray(a), np.asarray(b)))


def trans(x):
    return T(np.asarray(x).T)


def dot(a, b, allow_tf32=False):
    """Tensor-core matmul: BF16 operands, FP32 accumulate, no rounding until
    the caller casts."""
    return T(np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32))


def exp(x):
    return T(np.exp(np.asarray(x, dtype=np.float32)))


def static_range(*args):
    return range(*args)


class _Math:
    @staticmethod
    def exp2(x):
        return T(np.exp2(np.asarray(x, dtype=np.float32)))

    @staticmethod
    def rsqrt(x):
        return T(1.0 / np.sqrt(np.asarray(x, dtype=np.float32)))


math = _Math()


def constexpr(x):
    return x


# ---------------------------------------------------------------------------
# running a kernel
# ---------------------------------------------------------------------------


def run(module, kernel_name, grid, args, kwargs, jit_names=()):
    """Run ``module.<kernel_name>``'s source over ``grid``, on this shim.

    ``jit_names`` are other ``@triton.jit`` functions in the module that the
    kernel calls; they are unwrapped for the duration so a plain Python call
    reaches their source instead of Triton's launcher.
    """
    saved = {name: getattr(module, name) for name in ("tl",) + tuple(jit_names)}
    body = getattr(module, kernel_name)
    body = getattr(body, "fn", body)
    try:
        module.tl = _self_module()
        for name in jit_names:
            fn = getattr(module, name)
            setattr(module, name, getattr(fn, "fn", fn))
        grid = tuple(grid) + (1,) * (3 - len(grid))
        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    set_program((x, y, z), grid)
                    body(*args, **kwargs)
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _self_module():
    import sys
    return sys.modules[__name__]
