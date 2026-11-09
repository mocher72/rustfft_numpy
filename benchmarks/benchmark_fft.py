#!/usr/bin/env python3
"""Compare CPU FFTs and, when available, GPU-resident FFT implementations."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.fft
import rustfft_numpy as rustfft_np


SIZES = (1024, 8192, 32768, 65536, 262144, 1048576, 4194304, 16777216)
TARGET_SAMPLE_SECONDS = 0.20
MAX_CALLS_PER_SAMPLE = 1000
SAMPLE_COUNT = 5
BATCH_TARGET_ELEMENTS = 8_388_608


def machine_metadata() -> dict[str, str]:
    system = platform.system()
    cpu = platform.processor().strip()
    if system == "Linux":
        try:
            lines = subprocess.check_output(["lscpu"], text=True).splitlines()
            models = list(dict.fromkeys(
                line.split(":", 1)[-1].strip()
                for line in lines if line.lower().startswith("model name:")
            ))
            if models:
                cpu = " / ".join(models)
        except (OSError, subprocess.CalledProcessError):
            pass
    elif system == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    cpu = cpu or platform.machine() or "unknown-cpu"
    host = re.sub(r"[^A-Za-z0-9_.-]+", "-", platform.node()).strip("-") or "host"
    arch = platform.machine().lower() or "unknown-arch"
    slug = f"{system.lower()}-{arch}-{host}"
    return {
        "machine_id": slug,
        "os": platform.platform(),
        "cpu": cpu,
        "architecture": arch,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "docker_image": os.environ.get("BENCHMARK_DOCKER_IMAGE", ""),
    }


def measure(fn, data: np.ndarray) -> tuple[float, float, float, int]:
    fn(data)  # warm plan and allocator
    start = time.perf_counter()
    fn(data)
    one_call = time.perf_counter() - start
    calls = max(1, min(MAX_CALLS_PER_SAMPLE, int(TARGET_SAMPLE_SECONDS / max(one_call, 1e-6))))

    per_call_ms = []
    for _ in range(SAMPLE_COUNT):
        gc.collect()
        start = time.perf_counter()
        for _ in range(calls):
            result = fn(data)
        elapsed = time.perf_counter() - start
        del result
        per_call_ms.append(elapsed * 1000.0 / calls)
    return (
        statistics.median(per_call_ms),
        float(np.percentile(per_call_ms, 25)),
        float(np.percentile(per_call_ms, 75)),
        calls,
    )


def measure_gpu(fn, cupy) -> tuple[float, float, float, int]:
    """Time queued GPU work with CUDA events; all input data is already resident."""
    fn().get()  # warm the plan and ensure the first execution has completed

    def timed_batch(calls: int) -> float:
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        for _ in range(calls):
            result = fn()
        end.record()
        end.synchronize()
        del result
        return cupy.cuda.get_elapsed_time(start, end)

    one_call = timed_batch(1)
    calls = max(1, min(MAX_CALLS_PER_SAMPLE,
                       int(TARGET_SAMPLE_SECONDS / max(one_call / 1000.0, 1e-6))))
    per_call_ms = []
    for _ in range(SAMPLE_COUNT):
        gc.collect()
        per_call_ms.append(timed_batch(calls) / calls)
    return (
        statistics.median(per_call_ms),
        float(np.percentile(per_call_ms, 25)),
        float(np.percentile(per_call_ms, 75)),
        calls,
    )


def measure_torch_gpu(fn, torch) -> tuple[float, float, float, int]:
    """Measure PyTorch CUDA FFT work with CUDA events on its current stream."""
    fn()
    torch.cuda.synchronize()

    def timed_batch(calls: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            result = fn()
        end.record()
        end.synchronize()
        del result
        return start.elapsed_time(end)

    one_call = timed_batch(1)
    calls = max(1, min(MAX_CALLS_PER_SAMPLE,
                       int(TARGET_SAMPLE_SECONDS / max(one_call / 1000.0, 1e-6))))
    per_call_ms = []
    for _ in range(SAMPLE_COUNT):
        gc.collect()
        per_call_ms.append(timed_batch(calls) / calls)
    return (
        statistics.median(per_call_ms),
        float(np.percentile(per_call_ms, 25)),
        float(np.percentile(per_call_ms, 75)),
        calls,
    )


def torch_backend():
    """Return PyTorch and its CUDA device name when CUDA is available."""
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch benchmark unavailable: {exc}", flush=True)
        return None
    try:
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except (RuntimeError, OSError) as exc:
        print(f"PyTorch CUDA benchmark unavailable: {exc}", flush=True)
        device_name = None
    return torch, device_name


def vkfft_backend():
    """Return CuPy and its optional VkFFT entry point, or None without CUDA."""
    try:
        import cupy
        device_name = cupy.cuda.runtime.getDeviceProperties(
            cupy.cuda.Device().id
        )["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"CuPy CUDA benchmark unavailable: {exc}", flush=True)
        return None
    try:
        from pyvkfft.fft import fftn
        from pyvkfft.cuda import VkFFTApp  # confirm the CUDA extension is built
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"VkFFT GPU benchmark unavailable; CuPy cuFFT remains enabled: {exc}",
              flush=True)
        fftn = None
    return cupy, fftn, device_name


def benchmark_gpu_implementations(data: np.ndarray, cupy, vkfft_fftn,
                                  device_name: str, n: int, frames: int,
                                  workload: str) -> list[dict[str, object]]:
    # Transfer happens once, outside the timed region. FFT calls and their
    # outputs remain on the GPU, matching the common iterative GPU use case.
    gpu_data = cupy.asarray(data)
    implementations = []
    if vkfft_fftn is not None:
        implementations.append(
            ("VkFFT/CuPy (GPU resident)", lambda: vkfft_fftn(gpu_data, axes=(-1,)))
        )
    implementations.append(
        ("cuFFT/CuPy (GPU resident)", lambda: cupy.fft.fft(gpu_data, axis=-1))
    )
    rows = []
    for name, fn in implementations:
        med, q1, q3, calls = measure_gpu(fn, cupy)
        print(f"{workload} N={n:,} frames={frames} {name} [{device_name}]: "
              f"{med:9.4f} ms (Q1 {q1:.4f}, Q3 {q3:.4f}; "
              f"{calls} calls/sample; GPU-resident)", flush=True)
        rows.append({
            "workload": workload,
            "n": n,
            "frames": frames,
            "implementation": name,
            "device": device_name,
            "timing_scope": "device_resident",
            "median_ms": med,
            "q1_ms": q1,
            "q3_ms": q3,
            "calls_per_sample": calls,
            "samples": SAMPLE_COUNT,
        })
    del gpu_data
    return rows


def benchmark_torch(data: np.ndarray, torch, gpu_name: str | None,
                    n: int, frames: int, workload: str) -> list[dict[str, object]]:
    rows = []
    cpu_data = torch.from_numpy(data)
    cpu_fn = lambda _: torch.fft.fft(cpu_data, dim=-1)
    med, q1, q3, calls = measure(cpu_fn, data)
    rows.append({
        "workload": workload,
        "n": n,
        "frames": frames,
        "implementation": "torch.fft (CPU)",
        "device": "CPU",
        "timing_scope": "host_call",
        "median_ms": med,
        "q1_ms": q1,
        "q3_ms": q3,
        "calls_per_sample": calls,
        "samples": SAMPLE_COUNT,
    })
    print(f"{workload} N={n:,} frames={frames} torch.fft (CPU): "
          f"{med:9.4f} ms (Q1 {q1:.4f}, Q3 {q3:.4f}; "
          f"{calls} calls/sample; {torch.get_num_threads()} threads)", flush=True)

    if gpu_name is not None:
        # Upload once before timing so this measures resident PyTorch CUDA FFTs.
        gpu_data = torch.as_tensor(data, device="cuda")
        fn = lambda: torch.fft.fft(gpu_data, dim=-1)
        med, q1, q3, calls = measure_torch_gpu(fn, torch)
        rows.append({
            "workload": workload,
            "n": n,
            "frames": frames,
            "implementation": "torch.fft (CUDA/GPU resident)",
            "device": gpu_name,
            "timing_scope": "device_resident",
            "median_ms": med,
            "q1_ms": q1,
            "q3_ms": q3,
            "calls_per_sample": calls,
            "samples": SAMPLE_COUNT,
        })
        print(f"{workload} N={n:,} frames={frames} torch.fft (CUDA) [{gpu_name}]: "
              f"{med:9.4f} ms (Q1 {q1:.4f}, Q3 {q3:.4f}; "
              f"{calls} calls/sample; GPU-resident)", flush=True)
        del gpu_data
    return rows


def run(output: Path) -> None:
    rows = []
    metadata = machine_metadata()
    vkfft = vkfft_backend()
    torch_info = torch_backend()
    torch = torch_info[0] if torch_info else None
    torch_gpu_name = torch_info[1] if torch_info else None
    metadata["torch_version"] = torch.__version__ if torch is not None else ""
    metadata["torch_cpu_threads"] = str(torch.get_num_threads()) if torch is not None else ""
    metadata["cuda_version"] = (
        str(torch.version.cuda)
        if torch is not None and torch.version.cuda is not None
        else ""
    )
    implementations = (
        ("numpy.fft", np.fft.fft),
        ("scipy.fft", scipy.fft.fft),
        ("rustfft_numpy", rustfft_np.fft),
    )

    for n in SIZES:
        rng = np.random.default_rng(20261029 + n)
        data = (rng.standard_normal(n, dtype=np.float32) +
                1j * rng.standard_normal(n, dtype=np.float32)).astype(np.complex64)
        for name, fn in implementations:
            med, q1, q3, calls = measure(fn, data)
            rows.append({
                **metadata,
                "workload": "single_fft",
                "n": n,
                "frames": 1,
                "implementation": name,
                "device": "CPU",
                "timing_scope": "host_call",
                "median_ms": med,
                "q1_ms": q1,
                "q3_ms": q3,
                "calls_per_sample": calls,
                "samples": SAMPLE_COUNT,
            })
            print(f"single_fft N={n:>9,} {name:>13}: {med:9.4f} ms "
                  f"(Q1 {q1:.4f}, Q3 {q3:.4f}; {calls} calls/sample)", flush=True)
        if torch is not None:
            rows.extend({**metadata, **row} for row in benchmark_torch(
                data, torch, torch_gpu_name, n=n, frames=1, workload="single_fft"))
        if vkfft is not None:
            rows.extend({**metadata, **row} for row in benchmark_gpu_implementations(
                data, *vkfft, n=n, frames=1, workload="single_fft"))
        del data

    n = 32768
    frames = (90 * 96000 - n) // (n // 2) + 1
    rng = np.random.default_rng(20261029)
    data = (rng.standard_normal((frames, n), dtype=np.float32) +
            1j * rng.standard_normal((frames, n), dtype=np.float32)).astype(np.complex64)
    for name, fn in implementations:
        med, q1, q3, calls = measure(fn, data)
        rows.append({
            **metadata,
            "workload": "stft_frames_fft_only",
            "n": n,
            "frames": frames,
            "implementation": name,
            "device": "CPU",
            "timing_scope": "host_call",
            "median_ms": med,
            "q1_ms": q1,
            "q3_ms": q3,
            "calls_per_sample": calls,
            "samples": SAMPLE_COUNT,
        })
        print(f"stft_frames_fft_only N={n:,} frames={frames} {name:>13}: "
              f"{med:9.4f} ms (Q1 {q1:.4f}, Q3 {q3:.4f}; "
              f"{calls} calls/sample)", flush=True)
    if torch is not None:
        rows.extend({**metadata, **row} for row in benchmark_torch(
            data, torch, torch_gpu_name, n=n, frames=frames,
            workload="stft_frames_fft_only"))
    if vkfft is not None:
        rows.extend({**metadata, **row} for row in benchmark_gpu_implementations(
            data, *vkfft, n=n, frames=frames,
            workload="stft_frames_fft_only"))

    # Sweep batched 1D transforms while keeping each input batch near 64 MiB.
    # This amortizes Python and launch overhead for smaller FFT sizes.
    for n in SIZES:
        frames = max(1, BATCH_TARGET_ELEMENTS // n)
        rng = np.random.default_rng(20261101 + n)
        data = (rng.standard_normal((frames, n), dtype=np.float32) +
                1j * rng.standard_normal((frames, n), dtype=np.float32)).astype(np.complex64)
        for name, fn in implementations:
            med, q1, q3, calls = measure(fn, data)
            rows.append({
                **metadata,
                "workload": "batched_fft",
                "n": n,
                "frames": frames,
                "implementation": name,
                "device": "CPU",
                "timing_scope": "host_call",
                "median_ms": med,
                "q1_ms": q1,
                "q3_ms": q3,
                "calls_per_sample": calls,
                "samples": SAMPLE_COUNT,
            })
            print(f"batched_fft N={n:,} frames={frames} {name:>13}: "
                  f"{med:9.4f} ms total, {med / frames * 1000:8.3f} us/FFT "
                  f"(Q1 {q1:.4f}, Q3 {q3:.4f}; {calls} calls/sample)", flush=True)
        if torch is not None:
            rows.extend({**metadata, **row} for row in benchmark_torch(
                data, torch, torch_gpu_name, n=n, frames=frames,
                workload="batched_fft"))
        if vkfft is not None:
            rows.extend({**metadata, **row} for row in benchmark_gpu_implementations(
                data, *vkfft, n=n, frames=frames, workload="batched_fft"))
        del data

    for row in rows:
        row["per_fft_us"] = row["median_ms"] * 1000.0 / row["frames"]

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["machine_id", "os", "cpu", "architecture", "python", "numpy", "scipy",
                  "docker_image", "cuda_version", "torch_version", "torch_cpu_threads",
                  "workload", "n", "frames", "implementation", "device", "timing_scope", "median_ms",
                  "per_fft_us", "q1_ms", "q3_ms", "calls_per_sample", "samples"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output}")
    print(" | ".join(f"{key}: {value}" for key, value in metadata.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    default_name = f"results-{machine_metadata()['machine_id']}.csv"
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name(default_name))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
