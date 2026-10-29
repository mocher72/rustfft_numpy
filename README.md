# rustfft-numpy

A separate Python package that aims to mirror the `numpy.fft` API while using
RustFFT for its transforms. It returns NumPy arrays; it does not replace or
modify `numpy.fft`.

The compatibility target is NumPy 2.5.2, Python 3.12, and the
`separ` environment. The package uses PyO3 and rust-numpy to cross the Python / Rust
boundary. RustFFT plans are cached by transform length, direction, and precision.

## Build and benchmark on another machine

The Rust extension can be built natively on macOS Apple Silicon and Linux
x86_64 as well as this ARM64 Linux machine. Install Python 3.12, a stable Rust
toolchain, and `maturin`, then from the project root:

```sh
python -m pip install 'maturin>=1.8,<2' numpy scipy pandas matplotlib jupyter ipykernel
maturin develop --release
python benchmarks/benchmark_fft.py
jupyter notebook benchmarks/fft_benchmark.ipynb
```

Run the commands in an activated virtual environment. `maturin develop` builds
the extension for the current machine; no cross-compilation is needed for these
native runs. The benchmark writes a uniquely named
`benchmarks/results-<os>-<architecture>-<host>.csv` file and records CPU,
operating system, Python, NumPy, and SciPy metadata. Copy result CSVs from each
machine into the same `benchmarks` directory to compare them in one notebook.

Example:

```python
import numpy as np
import rustfft_numpy as rfft

x = np.ones(32768, dtype=np.complex64)
y = rfft.fft(x)
```

## Current implementation notes

- Complex transforms use RustFFT's `FftPlanner`, with reusable cached plans.
- The public Python layer covers the NumPy 2.5 FFT function names and signatures.
- Real transforms currently use a full complex transform and keep the required
  Hermitian half-spectrum. A `realfft` path can optimize this without changing
  the Python API.
- RustFFT supports `f32` and `f64`; NumPy extended precision (`longdouble` /
  `clongdouble`) is not implemented.
- API compatibility is under development. NumPy differential coverage is still
  needed before treating this package as a drop-in behavioral match.

## Building a macOS Apple Silicon wheel

On an Apple Silicon Mac with Python 3.12, Rust, and maturin installed, run from
this directory:

```sh
maturin build --release --interpreter "$(which python)" --out dist
```

This builds a native macOS ARM64 wheel. To install the package for local use,
run `maturin develop --release`, then run the benchmark with
`python benchmarks/benchmark_fft.py`. The notebook loads all host-specific
`benchmarks/results-*.csv` files in its directory; copy benchmark CSVs from
other machines there to plot them together.

Prebuilt Linux CPython 3.12 wheels for x86_64 and AArch64 are included under
`dist/`. They are tagged manylinux2014 / glibc 2.17 compatible.
