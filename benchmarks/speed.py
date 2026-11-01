"""Run times of opticalflow-torch against OpenCV.

Farneback: ``cv2.calcOpticalFlowFarneback`` on the CPU and on the GPU through OpenCL
(``cv2.UMat`` inputs) against ``farneback`` on the CPU and on CUDA, forward only and forward plus
backward, with OpenCV's defaults and with the settings used for radar ("livemap"), plus a
``torch.compile`` row whose compilation time is reported apart.
Lucas-Kanade: ``lucas_kanade`` against ``cv2.calcOpticalFlowPyrLK`` (CPU and OpenCL) given every
pixel as a point: the opencv-python wheels have no dense Lucas-Kanade
(``cv::cuda::DensePyrLKOpticalFlow`` needs a CUDA build with the contrib modules). These are
different algorithms: OpenCV's tracks points with a uniform window, Scharr gradients and early
stopping.

Frames are a smooth random texture in 0-255 and the same texture shifted by (2.5, -1.5) pixels,
float32 (uint8 for ``cv2.calcOpticalFlowPyrLK``). Inputs are on the device before timing; each
time is the median of repeated runs after warm-up, CUDA and OpenCL work synchronised. Batch rows
give the time per pair. Markdown tables go to stdout, progress to stderr.

    uv run python benchmarks/speed.py                  # full run, about 15 minutes (8 with --no-compile)
    uv run python benchmarks/speed.py --quick          # smoke test, tiny frames
    uv run python benchmarks/speed.py --json out.json  # also save the raw times and the environment
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import functools
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import opticalflow_torch as oft

SIZES = [(256, 256), (512, 512), (1024, 1024), (2200, 1900)]  # rows x columns; the last is OPERA's grid
QUICK_SIZES = [(64, 64), (96, 128)]
SHIFT = (2.5, -1.5)  # x, y, in pixels
BATCH = 8
CPU_WORKERS = torch.get_num_threads()  # parallel cv2 calls: as many as torch has CPU threads, one per physical core
CPU_BATCH_PIXELS = 512 * 512  # larger batches on the CPU would take minutes
FARNEBACK = {
    "OpenCV defaults": dict(pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1),
    "livemap": dict(pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2),
}
# OpenCV's defaults but at most 5 iterations per level, as lucas_kanade (winsize 21, levels 3, iterations 5);
# OpenCV's default of 30 takes about 13% longer. Also used by accuracy.py.
PYRLK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5, 0.01))
PYRLK_TEXT = "with winSize=(21, 21), maxLevel=3, criteria=(COUNT | EPS, 5, 0.01), uint8 frames"


@dataclass
class Timing:
    warmup: int  # calls before timing, one more on CUDA
    min_runs: int
    max_runs: int
    target: float  # seconds of timed runs per cell


@dataclass
class Result:
    method: str
    setting: str
    row: str
    height: int
    width: int
    batch: int
    times: list[float]  # seconds per call; empty if CUDA ran out of memory
    peak_mib: float | None = None  # CUDA memory allocated at the peak of a call, inputs included
    compile_s: float | None = None  # first call of the compiled function

    def per_pair(self) -> np.ndarray:
        return np.asarray(self.times) / self.batch


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def frame_pair(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A periodic smooth texture in 0-255 and the same shifted by ``SHIFT``, float32 ``(H, W)``.

    The shift is applied in Fourier space, so ``frame0[y, x] = frame1[y + SHIFT_y, x + SHIFT_x]``
    exactly, up to the final clipping.
    """
    noise = np.random.default_rng(seed).standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)
    spectrum = np.fft.rfft2(noise) * np.exp(-2 * (np.pi * 2.0) ** 2 * (fx**2 + fy**2))  # sigma 2 px
    shifted = spectrum * np.exp(-2j * np.pi * (fx * SHIFT[0] + fy * SHIFT[1]))
    frame0 = np.fft.irfft2(spectrum, s=(height, width))
    frame1 = np.fft.irfft2(shifted, s=(height, width))
    low, high = frame0.min(), frame0.max()
    return tuple(np.clip(255 * (f - low) / (high - low), 0, 255).astype(np.float32) for f in (frame0, frame1))


def to_tensor(frames: list[np.ndarray], device: str) -> torch.Tensor:
    """``(H, W)`` arrays as a ``(B, 1, H, W)`` tensor on ``device``."""
    return torch.from_numpy(np.stack(frames)[:, None]).to(device)


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------


def measure(fn: Callable[[], object], timing: Timing, sync: Callable[[], object] | None = None) -> list[float]:
    """Seconds per call of ``fn``: enough runs for ``timing.target`` seconds, within the bounds.

    ``sync`` waits for the device (CUDA or OpenCL) before and after each call.
    """

    def once() -> float:
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        return time.perf_counter() - start

    estimate = min(once() for _ in range(timing.warmup + (sync is not None)))
    runs = min(max(timing.min_runs, math.ceil(timing.target / estimate)), timing.max_runs)
    return [once() for _ in range(runs)]


def measure_cuda(fn: Callable[[], object], timing: Timing, setup: Callable[[], object] | None = None) -> dict:
    """``times`` and ``peak_mib`` of ``fn`` on CUDA, after ``setup`` (timed as ``compile_s``).

    Out of memory gives no times instead of stopping the run.
    """
    torch.cuda.empty_cache()
    out = {}
    try:
        if setup is not None:
            torch.cuda.synchronize()
            start = time.perf_counter()
            setup()
            torch.cuda.synchronize()
            out["compile_s"] = time.perf_counter() - start
        out["times"] = measure(fn, timing, sync=torch.cuda.synchronize)
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        out["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    except torch.cuda.OutOfMemoryError:
        out = {"times": []}
    torch.cuda.empty_cache()
    return out


def forward_backward(flow_fn: Callable, frame0: torch.Tensor, frame1: torch.Tensor) -> Callable[[], object]:
    """A call that computes the flow and its gradient with respect to ``frame0``."""
    frame0 = frame0.detach().requires_grad_(True)
    grad_flow = torch.randn(frame0.shape[0], 2, *frame0.shape[-2:], device=frame0.device)
    return lambda: torch.autograd.grad(flow_fn(frame0, frame1), frame0, grad_flow)


# ----------------------------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------------------------


def bench_farneback(
    setting: str, size: tuple[int, int], timing: Timing, cuda: bool, opencl: bool, compile_: bool
) -> list[Result]:
    params = FARNEBACK[setting]
    cv_args = tuple(params.values()) + (0,)  # flags=0: box window
    pairs = [frame_pair(*size, seed) for seed in range(max(BATCH, CPU_WORKERS))]
    batched = pairs[:BATCH]
    a, b = pairs[0]
    results = []

    def add(row: str, batch: int, times: list[float], **extra: float) -> None:
        results.append(Result("farneback", setting, row, *size, batch, times, **extra))
        took = f"{1e3 * np.median(times) / batch:.3g} ms per pair" if times else "out of memory"
        log(f"  {setting}, {size[0]}x{size[1]}, {row}, B={batch}: {took}")

    def cv_call(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(pair[0], pair[1], None, *cv_args)

    threads = cv2.getNumThreads()
    add("cv2, CPU", 1, measure(lambda: cv_call((a, b)), timing))
    cv2.setNumThreads(1)
    add("cv2, CPU, 1 thread", 1, measure(lambda: cv_call((a, b)), timing))
    cv2.setNumThreads(threads)
    add(f"cv2, CPU, loop of {BATCH} calls", BATCH, measure(lambda: [cv_call(p) for p in batched], timing))
    with concurrent.futures.ThreadPoolExecutor(CPU_WORKERS) as pool:  # cv2 releases the GIL
        times = measure(lambda: list(pool.map(cv_call, pairs[:CPU_WORKERS])), timing)
        add(f"cv2, CPU, {CPU_WORKERS} calls in {CPU_WORKERS} Python threads", CPU_WORKERS, times)
    if opencl:  # UMat inputs take OpenCV's OpenCL path; the flows stay on the device, as torch's do
        on_device = [(cv2.UMat(p[0]), cv2.UMat(p[1])) for p in batched]
        add("cv2, OpenCL", 1, measure(lambda: cv_call(on_device[0]), timing, sync=cv2.ocl.finish))
        times = measure(lambda: [cv_call(p) for p in on_device], timing, sync=cv2.ocl.finish)
        add(f"cv2, OpenCL, loop of {BATCH} calls", BATCH, times)

    flow_fn = functools.partial(oft.farneback, **params)
    with torch.inference_mode():
        f0, f1 = to_tensor([a], "cpu"), to_tensor([b], "cpu")
        add(f"farneback, CPU, {torch.get_num_threads()} threads", 1, measure(lambda: flow_fn(f0, f1), timing))
        if size[0] * size[1] <= CPU_BATCH_PIXELS:
            f0, f1 = to_tensor([p[0] for p in batched], "cpu"), to_tensor([p[1] for p in batched], "cpu")
            add(f"farneback, CPU, {torch.get_num_threads()} threads", BATCH, measure(lambda: flow_fn(f0, f1), timing))
    if not cuda:
        return results

    f0, f1 = to_tensor([a], "cuda"), to_tensor([b], "cuda")
    with torch.inference_mode():
        add("farneback, CUDA", 1, **measure_cuda(functools.partial(flow_fn, f0, f1), timing))
    add("farneback, CUDA, forward + backward", 1, **measure_cuda(forward_backward(flow_fn, f0, f1), timing))
    if compile_:
        torch._dynamo.reset()  # a fresh compilation per cell, and no recompilation limit
        run = functools.partial(torch.compile(flow_fn, fullgraph=True), f0, f1)
        with torch.inference_mode():
            add("farneback, CUDA, torch.compile", 1, **measure_cuda(run, timing, setup=run))
    f0, f1 = to_tensor([p[0] for p in batched], "cuda"), to_tensor([p[1] for p in batched], "cuda")
    with torch.inference_mode():
        add("farneback, CUDA", BATCH, **measure_cuda(functools.partial(flow_fn, f0, f1), timing))
    return results


def bench_lucas_kanade(size: tuple[int, int], timing: Timing, cuda: bool, opencl: bool) -> list[Result]:
    a, b = frame_pair(*size, seed=0)
    results = []

    def add(row: str, times: list[float], **extra: float) -> None:
        results.append(Result("lucas_kanade", "defaults", row, *size, 1, times, **extra))
        log(
            f"  Lucas-Kanade, {size[0]}x{size[1]}, {row}: "
            + (f"{1e3 * np.median(times):.3g} ms" if times else "out of memory")
        )

    a8, b8 = (np.rint(f).astype(np.uint8) for f in (a, b))  # cv2.calcOpticalFlowPyrLK takes 8-bit images only
    y, x = np.mgrid[: size[0], : size[1]]
    points = np.stack((x, y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    run = functools.partial(cv2.calcOpticalFlowPyrLK, a8, b8, points, None, **PYRLK)
    add(f"cv2.calcOpticalFlowPyrLK, every pixel, CPU, {cv2.getNumThreads()} threads", measure(run, timing))
    if opencl:
        run = functools.partial(cv2.calcOpticalFlowPyrLK, cv2.UMat(a8), cv2.UMat(b8), cv2.UMat(points), None, **PYRLK)
        add("cv2.calcOpticalFlowPyrLK, every pixel, OpenCL", measure(run, timing, sync=cv2.ocl.finish))

    with torch.inference_mode():
        f0, f1 = to_tensor([a], "cpu"), to_tensor([b], "cpu")
        add(f"lucas_kanade, CPU, {torch.get_num_threads()} threads", measure(lambda: oft.lucas_kanade(f0, f1), timing))
    if not cuda:
        return results
    f0, f1 = to_tensor([a], "cuda"), to_tensor([b], "cuda")
    with torch.inference_mode():
        add("lucas_kanade, CUDA", **measure_cuda(functools.partial(oft.lucas_kanade, f0, f1), timing))
    add("lucas_kanade, CUDA, forward + backward", **measure_cuda(forward_backward(oft.lucas_kanade, f0, f1), timing))
    return results


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------


def environment(threads: int, opencl: bool) -> dict[str, object]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = ""
    cuda = torch.cuda.is_available()
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": commit or None,
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "python": platform.python_version(),
        "opticalflow_torch": oft.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cv2": cv2.__version__,
        "cv2_opencl": cv2.ocl.Device.getDefault().name() if opencl else None,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "cv2_threads": threads,
    }


def fmt(seconds: float) -> str:
    ms = 1e3 * seconds
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def table(results: list[Result], sizes: list[tuple[int, int]], cell: Callable[[Result], str | None]) -> str:
    rows: dict[tuple[str, int], dict[tuple[int, int], str]] = {}
    for r in results:
        value = cell(r)
        if value is not None:
            rows.setdefault((r.row, r.batch), {})[(r.height, r.width)] = value
    if not rows:
        return ""
    header = "| | B | " + " | ".join(f"{h}x{w}" for h, w in sizes) + " |"
    lines = [header, "| --- | --- |" + " --- |" * len(sizes)]
    for (row, batch), cells in sorted(rows.items(), key=lambda item: item[0][1]):  # single pairs first
        lines.append(f"| {row} | {batch} | " + " | ".join(cells.get(s, "") for s in sizes) + " |")
    return "\n".join(lines)


def report(results: list[Result], sizes: list[tuple[int, int]], env: dict[str, object]) -> str:
    def time_cell(r: Result) -> str:
        if not r.times:
            return "out of memory"
        q25, q50, q75 = np.percentile(r.per_pair(), [25, 50, 75])
        return f"{fmt(q50)} ({fmt(q25)}-{fmt(q75)})"

    def speedup_cell(reference: dict[tuple[int, int], float]) -> Callable[[Result], str | None]:
        return lambda r: f"{reference[(r.height, r.width)] / np.median(r.per_pair()):.2g}x" if r.times else None

    out = [
        "# Run times, opticalflow-torch against OpenCV",
        "",
        f"{env['cpu']} ({env['logical_cpus']} logical CPUs), {env['gpu'] or 'no GPU'}; torch {env['torch']} "
        f"(CUDA {env['torch_cuda']}, {env['torch_threads']} CPU threads), OpenCV {env['cv2']} "
        f"({env['cv2_threads']} threads, OpenCL on {env['cv2_opencl'] or 'nothing'}), NumPy {env['numpy']}; "
        f"commit {env['commit']}, {env['date']}.",
        "",
        "Time per pair in ms: median (interquartile range). B is the number of pairs per call.",
    ]
    groups = [
        ("farneback", setting, f"Farneback, {setting}: " + ", ".join(f"{k}={v}" for k, v in params.items()))
        for setting, params in FARNEBACK.items()
    ]
    groups.append(
        ("lucas_kanade", "defaults", f"Lucas-Kanade: lucas_kanade defaults, cv2.calcOpticalFlowPyrLK {PYRLK_TEXT}")
    )
    for method, setting, title in groups:
        group = [r for r in results if r.method == method and r.setting == setting]
        if not group:
            continue
        out += ["", f"## {title}", "", table(group, sizes, time_cell)]
        cv_rows = [row for row in dict.fromkeys(r.row for r in group if r.batch == 1) if row.startswith("cv2")]
        for row in cv_rows[:1] + [row for row in cv_rows if "OpenCL" in row]:  # CPU, then OpenCL
            reference = {(r.height, r.width): float(np.median(r.per_pair())) for r in group if r.row == row and r.times}
            out += ["", f"Speed-up over {row}:", "", table(group, sizes, speedup_cell(reference))]
        memory = table(group, sizes, lambda r: None if r.peak_mib is None else f"{r.peak_mib:.0f}")
        if memory:
            out += ["", "CUDA peak memory, MiB, inputs included:", "", memory]
        compile_times = table(group, sizes, lambda r: None if r.compile_s is None else f"{r.compile_s:.1f}")
        if compile_times:
            out += ["", "torch.compile, first call (compilation), s:", "", compile_times]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny frames and 1-2 runs, for smoke tests")
    parser.add_argument("--json", type=Path, metavar="PATH", help="save the raw times and the environment")
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile rows")
    parser.add_argument("--cpu-only", action="store_true", help="skip the CUDA and OpenCL rows")
    args = parser.parse_args()

    sizes = QUICK_SIZES if args.quick else SIZES
    timing = Timing(1, 2, 2, 0.0) if args.quick else Timing(1, 5, 2000, 1.0)
    cuda = torch.cuda.is_available() and not args.cpu_only
    opencl = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL() and not args.cpu_only
    torch.manual_seed(0)
    if cuda and not args.no_compile:
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True  # report the time of a cold compilation

    results: list[Result] = []
    for setting in FARNEBACK:
        for size in sizes:
            results += bench_farneback(setting, size, timing, cuda, opencl, not args.no_compile and cuda)
    for size in sizes:
        results += bench_lucas_kanade(size, timing, cuda, opencl)

    env = environment(cv2.getNumThreads(), opencl)
    print(report(results, sizes, env), end="")
    if args.json:
        records = []
        for r in results:
            stats = {}
            if r.times:
                per_pair = r.per_pair()
                q25, q50, q75 = np.percentile(per_pair, [25, 50, 75])
                stats = {"median_s": q50, "q25_s": q25, "q75_s": q75, "min_s": per_pair.min()}
            records.append({**asdict(r), **{k: float(v) for k, v in stats.items()}})  # times per call, stats per pair
        options = {
            "quick": args.quick,
            "batch": BATCH,
            "cpu_workers": CPU_WORKERS,
            "shift": SHIFT,
            "farneback": FARNEBACK,
            "pyrlk": PYRLK_TEXT,
        }
        args.json.write_text(json.dumps({"environment": env, "options": options, "results": records}, indent=1))


if __name__ == "__main__":
    main()
