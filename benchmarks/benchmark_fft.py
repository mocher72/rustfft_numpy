#!/usr/bin/env python3
"""Compare NumPy, SciPy, and rustfft_numpy FFTs on complex64 inputs."""

from __future__ import annotations

import argparse
import csv
import gc
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


def run(output: Path) -> None:
    rows = []
    metadata = machine_metadata()
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
                "median_ms": med,
                "q1_ms": q1,
                "q3_ms": q3,
                "calls_per_sample": calls,
                "samples": SAMPLE_COUNT,
            })
            print(f"single_fft N={n:>9,} {name:>13}: {med:9.4f} ms "
                  f"(Q1 {q1:.4f}, Q3 {q3:.4f}; {calls} calls/sample)", flush=True)
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
            "median_ms": med,
            "q1_ms": q1,
            "q3_ms": q3,
            "calls_per_sample": calls,
            "samples": SAMPLE_COUNT,
        })
        print(f"stft_frames_fft_only N={n:,} frames={frames} {name:>13}: "
              f"{med:9.4f} ms (Q1 {q1:.4f}, Q3 {q3:.4f}; "
              f"{calls} calls/sample)", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["machine_id", "os", "cpu", "architecture", "python", "numpy", "scipy",
                  "workload", "n", "frames", "implementation", "median_ms",
                  "q1_ms", "q3_ms", "calls_per_sample", "samples"]
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
