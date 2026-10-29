"""NumPy 2.5 FFT-compatible Python layer over the RustFFT extension."""

from __future__ import annotations

import operator
import numpy as np

from . import _core


def _index(value, name):
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _norm(norm):
    if norm not in (None, "backward", "forward", "ortho"):
        raise ValueError('Invalid norm value; should be "backward", "ortho" or "forward"')
    return norm or "backward"


def _complex_dtype(dtype):
    result = np.dtype(np.result_type(dtype, 1j))
    if result == np.dtype(np.complex64):
        return np.dtype(np.complex64)
    if result == np.dtype(np.complex128):
        return np.dtype(np.complex128)
    raise TypeError(f"RustFFT backend does not support NumPy FFT output dtype {result}")


def _real_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        return np.dtype(np.float64)
    if dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        return dtype
    raise TypeError(f"RustFFT backend does not support NumPy FFT input dtype {dtype}")


def _resize(x, axis, n):
    n = _index(n, "n")
    if n == -1:
        return x
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    old = x.shape[axis]
    if old == n:
        return x
    shape = list(x.shape)
    shape[axis] = n
    out = np.zeros(shape, dtype=x.dtype)
    take = min(old, n)
    if take:
        src = [slice(None)] * x.ndim
        dst = [slice(None)] * x.ndim
        src[axis] = slice(0, take)
        dst[axis] = slice(0, take)
        out[tuple(dst)] = x[tuple(src)]
    return out


def _axis_fft(x, axis, inverse, norm):
    n = x.shape[axis]
    moved = np.ascontiguousarray(np.moveaxis(x, axis, -1))
    batch = moved.reshape((-1, n))
    if x.dtype == np.dtype(np.complex64):
        transformed = _core.fft32(batch, inverse)
    else:
        transformed = _core.fft64(batch, inverse)
    result = np.moveaxis(transformed.reshape(moved.shape), -1, axis)
    mode = _norm(norm)
    if mode == "ortho":
        result *= 1.0 / np.sqrt(n)
    elif (mode == "forward" and not inverse) or (mode == "backward" and inverse):
        result *= 1.0 / n
    return result


def _axes_and_sizes(ndim, shape, axes, default_axes=None):
    if axes is None:
        if shape is not None:
            shape = tuple(shape) if np.iterable(shape) else (shape,)
            if default_axes is None:
                axes = tuple(range(ndim - len(shape), ndim))
            else:
                axes = tuple(default_axes)
        elif default_axes is not None:
            axes = tuple(default_axes)
        else:
            axes = tuple(range(ndim))
    else:
        axes = tuple(axes) if np.iterable(axes) else (axes,)

    normalized = []
    for axis in axes:
        axis = _index(axis, "axis")
        if axis < -ndim or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim=ndim)
        axis %= ndim
        if axis in normalized:
            raise ValueError("all axes must be unique")
        normalized.append(axis)

    if shape is None:
        sizes = None
    else:
        shape = tuple(shape) if np.iterable(shape) else (shape,)
        if len(shape) != len(normalized):
            raise ValueError("Shape and axes have different lengths")
        sizes = tuple(_index(n, "s") for n in shape)
    return tuple(normalized), sizes


def _complex_nd(a, shape, axes, inverse, norm):
    x = np.asarray(a)
    axes, sizes = _axes_and_sizes(x.ndim, shape, axes)
    dtype = _complex_dtype(x.dtype)
    x = x.astype(dtype, copy=False)
    for i, axis in enumerate(axes):
        n = x.shape[axis] if sizes is None else sizes[i]
        x = _resize(x, axis, n)
        x = _axis_fft(x, axis, inverse, norm)
    return x


def _return(result, out):
    if out is None:
        return result
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a numpy.ndarray")
    if out.shape != result.shape:
        raise ValueError("output array has wrong shape")
    np.copyto(out, result, casting="same_kind")
    return out


def fft(a, n=None, axis=-1, norm=None, out=None):
    x = np.asarray(a)
    axis = _index(axis, "axis")
    if axis < -x.ndim or axis >= x.ndim:
        raise np.exceptions.AxisError(axis, ndim=x.ndim)
    axis %= x.ndim
    if n is None:
        n = x.shape[axis]
    return _return(_complex_nd(x, (n,), (axis,), False, norm), out)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    x = np.asarray(a)
    axis = _index(axis, "axis")
    if axis < -x.ndim or axis >= x.ndim:
        raise np.exceptions.AxisError(axis, ndim=x.ndim)
    axis %= x.ndim
    if n is None:
        n = x.shape[axis]
    return _return(_complex_nd(x, (n,), (axis,), True, norm), out)


def fftn(a, s=None, axes=None, norm=None, out=None):
    return _return(_complex_nd(a, s, axes, False, norm), out)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    return _return(_complex_nd(a, s, axes, True, norm), out)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _return(_complex_nd(a, s, axes, False, norm), out)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _return(_complex_nd(a, s, axes, True, norm), out)


def rfft(a, n=None, axis=-1, norm=None, out=None):
    x = np.asarray(a)
    if x.dtype.kind == "c":
        raise TypeError("rfft input must be real")
    axis = _index(axis, "axis")
    if axis < -x.ndim or axis >= x.ndim:
        raise np.exceptions.AxisError(axis, ndim=x.ndim)
    axis %= x.ndim
    if n is None:
        n = x.shape[axis]
    n = _index(n, "n")
    x = x.astype(_real_dtype(x.dtype), copy=False)
    x = _resize(x, axis, n)
    ctype = np.dtype(np.complex64 if x.dtype == np.dtype(np.float32) else np.complex128)
    z = x.astype(ctype)
    z = _axis_fft(z, axis, False, norm)
    sl = [slice(None)] * z.ndim
    sl[axis] = slice(0, n // 2 + 1)
    return _return(z[tuple(sl)], out)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    x = np.asarray(a)
    axis = _index(axis, "axis")
    if axis < -x.ndim or axis >= x.ndim:
        raise np.exceptions.AxisError(axis, ndim=x.ndim)
    axis %= x.ndim
    if n is None:
        n = 2 * (x.shape[axis] - 1)
    n = _index(n, "n")
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    ctype = _complex_dtype(x.dtype)
    x = x.astype(ctype, copy=False)
    half = n // 2 + 1
    x = _resize(x, axis, half)
    full_shape = list(x.shape)
    full_shape[axis] = n
    full = np.zeros(full_shape, dtype=ctype)
    idx_src = [slice(None)] * x.ndim
    idx_dst = [slice(None)] * full.ndim
    idx_src[axis] = slice(0, half)
    idx_dst[axis] = slice(0, half)
    full[tuple(idx_dst)] = x[tuple(idx_src)]
    stop = (n - 1) // 2 + 1
    for k in range(1, stop):
        src = [slice(None)] * full.ndim
        dst = [slice(None)] * full.ndim
        src[axis] = k
        dst[axis] = n - k
        full[tuple(dst)] = np.conjugate(full[tuple(src)])
    real = _axis_fft(full, axis, True, norm).real
    return _return(real, out)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    x = np.asarray(a)
    if x.dtype.kind == "c":
        raise TypeError("rfftn input must be real")
    axes, sizes = _axes_and_sizes(x.ndim, s, axes)
    if not axes:
        return _return(x.copy(), out)
    x = x.astype(_real_dtype(x.dtype), copy=False)
    resolved = tuple(x.shape[ax] if sizes is None or sizes[i] == -1 else sizes[i]
                     for i, ax in enumerate(axes))
    for i, axis in enumerate(axes):
        x = _resize(x, axis, resolved[i])
    ctype = np.dtype(np.complex64 if x.dtype == np.dtype(np.float32) else np.complex128)
    z = x.astype(ctype)
    # Applying the full complex transform and retaining the nonredundant
    # half-spectrum gives NumPy's rfftn convention; realfft can optimize this
    # path later without changing the Python API.
    for axis, n in zip(axes, resolved):
        z = _axis_fft(z, axis, False, norm)
    axis = axes[-1]
    sl = [slice(None)] * z.ndim
    sl[axis] = slice(0, resolved[-1] // 2 + 1)
    return _return(z[tuple(sl)], out)


def irfftn(a, s=None, axes=None, norm=None, out=None):
    x = np.asarray(a)
    axes, sizes = _axes_and_sizes(x.ndim, s, axes)
    if not axes:
        return _return(x.copy(), out)
    if sizes is None:
        resolved = [x.shape[ax] for ax in axes]
        resolved[-1] = 2 * (x.shape[axes[-1]] - 1)
    else:
        resolved = [x.shape[ax] if n == -1 else n for ax, n in zip(axes, sizes)]
    if any(n < 1 for n in resolved):
        raise ValueError("Invalid number of FFT data points specified")
    ctype = _complex_dtype(x.dtype)
    z = x.astype(ctype, copy=False)
    real_axis = axes[-1]
    nreal = resolved[-1]
    half = nreal // 2 + 1
    z = _resize(z, real_axis, half)
    for axis, n in zip(axes[:-1], resolved[:-1]):
        z = _resize(z, axis, n)
    full_shape = list(z.shape)
    full_shape[real_axis] = nreal
    full = np.zeros(full_shape, dtype=ctype)
    src = [slice(None)] * z.ndim
    dst = [slice(None)] * z.ndim
    src[real_axis] = slice(0, half)
    dst[real_axis] = slice(0, half)
    full[tuple(dst)] = z[tuple(src)]
    for k in range(1, (nreal - 1) // 2 + 1):
        src = [slice(None)] * full.ndim
        dst = [slice(None)] * full.ndim
        src[real_axis] = k
        dst[real_axis] = nreal - k
        full[tuple(dst)] = np.conjugate(full[tuple(src)])
    for axis in axes:
        full = _axis_fft(full, axis, True, norm)
    return _return(full.real, out)


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return rfftn(a, s, axes, norm, out)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return irfftn(a, s, axes, norm, out)


def _swap_norm(norm):
    norm = _norm(norm)
    return {"backward": "forward", "forward": "backward", "ortho": "ortho"}[norm]


def hfft(a, n=None, axis=-1, norm=None, out=None):
    return _return(irfft(np.conjugate(a), n, axis, _swap_norm(norm)), out)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    return _return(np.conjugate(rfft(np.conjugate(a), n, axis, _swap_norm(norm))), out)


def fftfreq(n, d=1.0, device=None):
    if device not in (None, "cpu"):
        raise ValueError("only device='cpu' is supported")
    n = _index(n, "n")
    if n < 0:
        raise ValueError("n should be nonnegative")
    return np.fft.fftfreq(n, d=d, device=device)


def rfftfreq(n, d=1.0, device=None):
    if device not in (None, "cpu"):
        raise ValueError("only device='cpu' is supported")
    n = _index(n, "n")
    if n < 0:
        raise ValueError("n should be nonnegative")
    return np.fft.rfftfreq(n, d=d, device=device)


def fftshift(x, axes=None):
    return np.fft.fftshift(x, axes=axes)


def ifftshift(x, axes=None):
    return np.fft.ifftshift(x, axes=axes)
