"""NumPy 2.5-style FFT functions backed by RustFFT.

The API aims to mirror ``numpy.fft`` while remaining a separate import.
"""

from ._api import (
    fft, ifft, fft2, ifft2, fftn, ifftn,
    rfft, irfft, rfft2, irfft2, rfftn, irfftn,
    hfft, ihfft, fftfreq, rfftfreq, fftshift, ifftshift,
)

__all__ = [
    "fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
    "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn",
    "hfft", "ihfft", "fftfreq", "rfftfreq", "fftshift", "ifftshift",
]
