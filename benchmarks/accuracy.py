"""Accuracy of opticalflow-torch against OpenCV.

Synthetic pairs with exact flow: interior endpoint error of Farneback (OpenCV and the port, same settings),
dense Lucas-Kanade and OpenCV's sparse ``calcOpticalFlowPyrLK``, the distance between the two Farneback
flows, and how large a uniform motion each method follows. OPERA radar triplets: the middle frame
interpolated from the outer two, scored against the real one. Prints Markdown tables.

    uv run python benchmarks/accuracy.py [--quick] [--json PATH] [--device cpu]
    uv run --with h5py python benchmarks/accuracy.py --opera-root /disks/fast/openradar

The radar part needs h5py and the OPERA ODYSSEY archive; it is skipped when either is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synthetic import AnalyticTexture, make_pair, rotation_flow, translation_flow  # noqa: E402

from opticalflow_torch import endpoint_error, farneback, interpolate_frames, lucas_kanade  # noqa: E402

# in the order of calcOpticalFlowFarneback's positional arguments
FARNEBACK = {
    "defaults": dict(pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1),
    "livemap": dict(pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2),
}
BORDER = 32  # band excluded from interior scores, in pixels
INNER = 96  # band excluded from the second interior maximum of the Farneback parity, in pixels
DISPLACEMENTS = {"small": 1.5, "medium": 5.0, "large": 18.0}  # largest interior displacement, in pixels
NOISE = 4.0  # standard deviation of the noise case, in 0-255 units
GRID_STEP = 16  # spacing of the points tracked by calcOpticalFlowPyrLK
# as in speed.py: OpenCV's defaults but at most 5 iterations per level, as lucas_kanade
PYRLK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5, 0.01))
SWEEP = [8.0, 16.0, 24.0, 32.0, 48.0]  # uniform translations, in pixels
SWEEP_WAVELENGTHS = [64.0, 256.0]  # longest wavelength of the texture (the tests' default, and longer)

# First frame of each OPERA triplet (t0, t0 + 15 min, t0 + 30 min): two rainy days per season, one per year,
# with the same radar coverage in the three frames
TRIPLETS = [
    "2014-02-01T12:00",
    "2020-02-16T13:00",
    "2015-05-19T12:00",
    "2021-05-16T12:00",
    "2016-06-23T14:00",
    "2018-08-25T13:00",
    "2013-11-04T13:00",
    "2023-10-19T12:00",
]
DBZ_MAX = 60.0  # frames are 0-60 dBZ scaled to 0-255
CSI_DBZ = 20.0
# lucas_kanade for radar: the lowest mean MAE over all covered pixels on five other triplets (12:00 UTC on
# 2017-04-25, 2019-05-28, 2019-11-28, 2022-01-04, 2024-02-22), among levels 3-5 and winsize 21-101
LUCAS_KANADE_RADAR = dict(levels=4, winsize=51)
PAD = 128  # zero margin that keeps the radar flows' coarsest levels away from the image edges, in pixels


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "–" if math.isnan(value) else f"{value:.3g}" if abs(value) < 1e-2 else f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def to_cv(frame: Tensor) -> np.ndarray:
    """``(1, 1, H, W)`` frame as a float32 ``(H, W)`` array."""
    return np.ascontiguousarray(frame[0, 0].detach().cpu().numpy().astype(np.float32))


def cv_farneback(frame0: Tensor, frame1: Tensor, settings: dict) -> Tensor:
    """``cv2.calcOpticalFlowFarneback`` on float32 frames, as a ``(1, 2, H, W)`` flow on the frames' device."""
    flow = cv2.calcOpticalFlowFarneback(to_cv(frame0), to_cv(frame1), None, *settings.values(), 0)
    return torch.from_numpy(flow).permute(2, 0, 1)[None].to(frame0)


def interior(x: Tensor, border: int = BORDER) -> Tensor:
    return x[..., border:-border, border:-border]


def difference(flow: Tensor, reference: Tensor) -> dict[str, float]:
    """Mean and max endpoint distance between two flows over the image, max over the interior and further in."""
    distance = endpoint_error(flow, reference)
    return {
        "mean": distance.mean().item(),
        "max": distance.max().item(),
        "max_interior": interior(distance).max().item(),
        "max_inner": interior(distance, INNER).max().item(),
    }


def nonrigid_flow(size: int) -> Tensor:
    """A drift with smooth shear and stretching, ``(1, 2, H, W)`` float64, largest interior displacement 1."""
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float64), torch.arange(size, dtype=torch.float64), indexing="ij"
    )
    fx = 0.7 + 0.3 * torch.sin(2 * math.pi * y / 256 + 0.4)
    fy = -0.5 + 0.4 * torch.cos(2 * math.pi * x / 320 + 1.1) * torch.cos(2 * math.pi * y / 400)
    flow = torch.stack((fx, fy))[None]
    return flow / interior(flow.norm(dim=1)).max()


def synthetic_cases(size: int) -> list[dict]:
    """Name, exact flow (float64) and noise of each synthetic case."""
    radius = math.sqrt(2) * ((size - 1) / 2 - BORDER)  # centre to interior corner
    dx, dy = math.cos(math.radians(33)), -math.sin(math.radians(33))  # not a whole number of pixels
    cases = []
    for label, d in DISPLACEMENTS.items():
        cases += [
            {
                "name": f"translation, {label}",
                "flow": translation_flow(size, size, dx * d, dy * d, dtype=torch.float64),
            },
            {
                "name": f"rotation, {label}",
                "flow": rotation_flow(size, size, math.degrees(d / radius), dtype=torch.float64),
            },
            {"name": f"non-rigid, {label}", "flow": d * nonrigid_flow(size)},
        ]
    d = DISPLACEMENTS["medium"]
    cases.append({"name": "non-rigid, medium, noise", "flow": d * nonrigid_flow(size), "noise": NOISE})
    return cases


def to_uint8(frame: Tensor) -> np.ndarray:
    return np.clip(np.rint(to_cv(frame)), 0, 255).astype(np.uint8)


def grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of points every ``GRID_STEP`` pixels in the interior."""
    ys, xs = np.mgrid[BORDER : size - BORDER + 1 : GRID_STEP, BORDER : size - BORDER + 1 : GRID_STEP]
    return ys.ravel(), xs.ravel()


def pyrlk(image0: np.ndarray, image1: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.calcOpticalFlowPyrLK`` from the points ``(xs, ys)``: displacements ``(N, 2)``, x first, and status."""
    points = np.stack((xs, ys), axis=-1).astype(np.float32)[:, None]
    moved, status, _ = cv2.calcOpticalFlowPyrLK(image0, image1, points, None, **PYRLK)
    return moved[:, 0] - points[:, 0], status.ravel() == 1


def sparse_lk(frame0: Tensor, frame1: Tensor, truth: Tensor) -> dict[str, float]:
    """``cv2.calcOpticalFlowPyrLK`` on a grid of points and ``lucas_kanade`` there, on uint8 frames.

    Errors are averaged over the points OpenCV tracks (status 1).
    """
    image0, image1 = to_uint8(frame0), to_uint8(frame1)
    ys, xs = grid(image0.shape[0])
    displacement, tracked = pyrlk(image0, image1, ys, xs)
    true = truth[0, :, ys, xs].T.cpu().numpy()  # (N, 2), x first
    cv_error = np.linalg.norm(displacement - true, axis=-1)
    to_torch = lambda image: torch.from_numpy(image).to(frame0)[None, None]  # noqa: E731
    dense = lucas_kanade(to_torch(image0), to_torch(image1))
    lk_error = np.linalg.norm(dense[0, :, ys, xs].T.cpu().numpy() - true, axis=-1)
    return {
        "points": len(ys),
        "lost": int((~tracked).sum()),
        "cv2": float(cv_error[tracked].mean()),
        "lucas_kanade": float(lk_error[tracked].mean()),
        "lucas_kanade_all": float(lk_error.mean()),
    }


def run_synthetic(size: int, device: torch.device) -> tuple[str, dict]:
    texture = AnalyticTexture(seed=0)
    rng = np.random.default_rng(0)
    methods = ["cv2 defaults", "farneback defaults", "cv2 livemap", "farneback livemap", "lucas_kanade"]
    results = []
    for case in synthetic_cases(size):
        log(f"synthetic: {case['name']}")
        truth = case["flow"]
        frame0, frame1 = make_pair(texture, truth)
        if case.get("noise"):
            frame0 = frame0 + torch.from_numpy(rng.normal(0, case["noise"], frame0.shape))
            frame1 = frame1 + torch.from_numpy(rng.normal(0, case["noise"], frame1.shape))
        frame0, frame1 = frame0.to(device, torch.float32), frame1.to(device, torch.float32)
        truth = truth.to(device, torch.float32)
        flows = {}
        for name, settings in FARNEBACK.items():
            flows[f"cv2 {name}"] = cv_farneback(frame0, frame1, settings)
            flows[f"farneback {name}"] = farneback(frame0, frame1, **settings)
        flows["lucas_kanade"] = lucas_kanade(frame0, frame1)
        speed = interior(truth.norm(dim=1))
        errors = {m: interior(endpoint_error(flows[m], truth)) for m in methods}
        results.append(
            {
                "case": case["name"],
                "mean_displacement": speed.mean().item(),
                "max_displacement": speed.max().item(),
                "epe": {m: e.mean().item() for m, e in errors.items()},
                "epe_p95": {m: e.quantile(0.95).item() for m, e in errors.items()},
                "parity": {n: difference(flows[f"farneback {n}"], flows[f"cv2 {n}"]) for n in FARNEBACK},
                "sparse": sparse_lk(frame0, frame1, truth),
            }
        )

    statistics = ("mean", "max", "max_interior", "max_inner")
    epe_rows = [
        [r["case"], f"{r['mean_displacement']:.1f} / {r['max_displacement']:.1f}", *r["epe"].values()] for r in results
    ]
    parity_rows = [[r["case"], *(r["parity"][n][k] for n in FARNEBACK for k in statistics)] for r in results]
    sparse_rows = [[r["case"], *r["sparse"].values()][:5] for r in results]
    out = [
        f"## Synthetic pairs ({size} x {size})",
        "",
        "Frames from `tests/synthetic.py` (AnalyticTexture, values 0-255) with exact flows. Displacement is the "
        f"mean and the largest flow length in the interior; scores exclude a {BORDER}-pixel border band. The "
        f"noise case adds Gaussian noise of standard deviation {NOISE:g} to each frame.",
        "",
        "### Mean interior endpoint error (pixels)",
        "",
        "Farneback with the OpenCV defaults and with the livemap settings (levels 3, winsize 25, iterations 3, "
        "poly_sigma 1.2); `lucas_kanade` with its defaults. OpenCV gets the same float32 frames.",
        "",
        table(["case", "displacement", *(f"`{m}`" for m in methods)], epe_rows),
        "",
        "### Farneback port against OpenCV (endpoint distance, pixels)",
        "",
        "Distance between the flows of `farneback` and `cv2.calcOpticalFlowFarneback` with the same settings: "
        f"mean and max over the image, max beyond {BORDER} and beyond {INNER} pixels from the edges. The two "
        "differ by design within about a pixel of displaced positions leaving the image (see the Farneback "
        "page), and at coarse pyramid levels the window spreads this inwards.",
        "",
        table(
            [
                "case",
                *(f"{n} {k}" for n in FARNEBACK for k in ("mean", "max", f"max >{BORDER} px", f"max >{INNER} px")),
            ],
            parity_rows,
        ),
        "",
        "### Sparse Lucas-Kanade (mean endpoint error at grid points, pixels)",
        "",
        f"`cv2.calcOpticalFlowPyrLK` (21 x 21 window, maxLevel 3, at most 5 iterations) at points every {GRID_STEP} "
        "pixels in the interior, on the frames rounded to uint8 as it requires; `lucas_kanade` (defaults) on the "
        "same uint8 frames, read at the same points. Errors over the points OpenCV tracks.",
        "",
        table(["case", "points", "lost by cv2", "`calcOpticalFlowPyrLK`", "`lucas_kanade`"], sparse_rows),
    ]
    return "\n".join(out), {"size": size, "border": BORDER, "noise": NOISE, "cases": results}


def run_sweep(size: int, device: torch.device) -> tuple[str, dict]:
    """Mean endpoint error at grid points for uniform translations of growing length, on two textures."""
    methods = {
        "farneback defaults": lambda a, b: farneback(a, b, **FARNEBACK["defaults"]),
        "farneback livemap": lambda a, b: farneback(a, b, **FARNEBACK["livemap"]),
        "lucas_kanade": lucas_kanade,
        "lucas_kanade levels=5": lambda a, b: lucas_kanade(a, b, levels=5),
    }
    dx, dy = math.cos(math.radians(33)), -math.sin(math.radians(33))
    ys, xs = grid(size)
    results, out = [], []
    for wavelength in SWEEP_WAVELENGTHS:
        texture = AnalyticTexture(seed=0, max_wavelength=wavelength)
        rows = []
        for d in SWEEP:
            log(f"sweep: wavelength {wavelength:g}, {d:g} px")
            truth = translation_flow(size, size, dx * d, dy * d, dtype=torch.float64)
            frame0, frame1 = (f.to(device, torch.float32) for f in make_pair(texture, truth))
            true = np.array([dx * d, dy * d])
            errors = {}
            for name, method in methods.items():
                flow = method(frame0, frame1)[0, :, ys, xs].T.cpu().numpy()
                errors[name] = float(np.linalg.norm(flow - true, axis=-1).mean())
            displacement, _ = pyrlk(to_uint8(frame0), to_uint8(frame1), ys, xs)
            errors["cv2.calcOpticalFlowPyrLK"] = float(np.linalg.norm(displacement - true, axis=-1).mean())
            results.append({"max_wavelength": wavelength, "displacement": d, "epe": errors})
            rows.append([f"{d:g}", *errors.values()])
        names = [*methods, "cv2.calcOpticalFlowPyrLK"]
        out += [
            "",
            f"Longest wavelength {wavelength:g} pixels:",
            "",
            table(["shift (px)", *(f"`{m}`" for m in names)], rows),
        ]
    head = [
        "### Uniform translations of growing length (mean endpoint error at grid points, pixels)",
        "",
        f"Translations at 33 degrees on {size} x {size} frames from AnalyticTexture, with wavelengths from 8 pixels "
        "to the longest given: how far each method follows a motion depends on the texture as well as on its "
        "pyramid. Errors at all the grid points above; `cv2.calcOpticalFlowPyrLK` as above, on uint8 frames.",
    ]
    return "\n".join(head + out), {"size": size, "results": results}


def opera_path(root: Path, time: datetime) -> Path:
    return root / f"{time:%Y/%m/%d}/OPERA/COMP/OPERA@{time:%Y%m%dT%H%M}@0@DBZH_QIND.h5"


def load_opera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reflectivity clipped to 0-60 dBZ and scaled to 0-255 (no echo and no data as 0), and the coverage mask."""
    import h5py

    with h5py.File(path) as f:
        what = dict(f["dataset1/what"].attrs)
        raw = f["dataset1/data1/data"][()].astype(np.float64)
    dbz = raw * what["gain"] + what["offset"]
    nodata = raw == what["nodata"]
    dbz[nodata | (raw == what["undetect"])] = 0.0
    return (np.clip(dbz, 0, DBZ_MAX) * 255 / DBZ_MAX).astype(np.float32), ~nodata


def padded_difference(frame0: Tensor, frame1: Tensor, settings: dict) -> dict[str, float]:
    """``difference`` of the Farneback flows of the frames with ``PAD`` zero pixels added around them."""
    frame0, frame1 = (torch.nn.functional.pad(f, (PAD,) * 4) for f in (frame0, frame1))
    flow, reference = farneback(frame0, frame1, **settings), cv_farneback(frame0, frame1, settings)
    return difference(interior(flow, PAD), interior(reference, PAD))


def scores(predicted: Tensor, observed: Tensor, mask: Tensor) -> dict[str, float]:
    """MAE, RMSE, bias and CSI at ``CSI_DBZ`` in dBZ over ``mask``."""
    p, o = predicted[mask].double() * DBZ_MAX / 255, observed[mask].double() * DBZ_MAX / 255
    error = p - o
    hits = ((p >= CSI_DBZ) & (o >= CSI_DBZ)).sum().item()
    wrong = ((p >= CSI_DBZ) != (o >= CSI_DBZ)).sum().item()
    return {
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "bias": error.mean().item(),
        "csi": hits / max(hits + wrong, 1),
    }


def run_radar(root: Path, triplets: list[str], device: torch.device) -> tuple[str, dict]:
    settings = FARNEBACK["livemap"]
    methods = ["cv2 Farneback", "farneback", "lucas_kanade", "lucas_kanade, radar", "zero flow", "persistence"]
    results = []
    for start in triplets:
        log(f"radar: {start}")
        t0 = datetime.fromisoformat(start)
        loaded = [load_opera(opera_path(root, t0 + timedelta(minutes=15 * k))) for k in range(3)]
        frame0, middle, frame1 = (torch.from_numpy(f).to(device)[None, None] for f, _ in loaded)
        covered = torch.from_numpy(np.logical_and.reduce([c for _, c in loaded])).to(device)
        wet = covered & (torch.maximum(torch.maximum(frame0, middle), frame1)[0, 0] > 0)
        flows = {
            "cv2 Farneback": cv_farneback(frame0, frame1, settings),
            "farneback": farneback(frame0, frame1, **settings),
            "lucas_kanade": lucas_kanade(frame0, frame1),
            "lucas_kanade, radar": lucas_kanade(frame0, frame1, **LUCAS_KANADE_RADAR),
            "zero flow": torch.zeros_like(torch.cat((frame0, frame0), dim=1)),
        }
        speed = flows["cv2 Farneback"].norm(dim=1)[0][wet]
        entry = {
            "t0": f"{t0:%Y-%m-%d %H:%M}",
            "covered_fraction": covered.double().mean().item(),
            "wet_fraction": (wet.sum() / covered.sum()).item(),
            "mean_displacement": speed.mean().item(),
            "p95_displacement": float(np.percentile(speed.cpu().numpy(), 95)),
            "parity": difference(flows["farneback"], flows["cv2 Farneback"]),
            "parity_padded": padded_difference(frame0, frame1, settings),
            "scores": {},
            "mae_observed": {},
        }
        predicted = {"persistence": frame0[0, 0]}
        for name, flow in flows.items():
            for matched in (False, True):
                frame = interpolate_frames(frame0, frame1, flow, 0.5, match_distribution=matched)[0, 0, 0]
                predicted[f"{name}{', matched' if matched else ''}"] = frame
        # rain predicted where none was observed counts as well: one mask for every method
        scored = wet | (covered & (torch.stack(list(predicted.values())).amax(dim=0) > 0))
        entry["scored_fraction"] = (scored.sum() / covered.sum()).item()
        for name, frame in predicted.items():
            entry["scores"][name] = scores(frame, middle[0, 0], scored)
            entry["mae_observed"][name] = scores(frame, middle[0, 0], wet)["mae"]
        results.append(entry)

    rows = [m + (", matched" if matched else "") for m in methods[:-1] for matched in (False, True)] + ["persistence"]
    mean = {
        r: {
            **{k: float(np.mean([e["scores"][r][k] for e in results])) for k in ("mae", "rmse", "bias", "csi")},
            "mae_observed": float(np.mean([e["mae_observed"][r] for e in results])),
        }
        for r in rows
    }
    triplet_rows = [
        [
            e["t0"],
            f"{e['covered_fraction']:.1%}",
            f"{e['wet_fraction']:.1%}",
            f"{e['scored_fraction']:.1%}",
            e["mean_displacement"],
            e["p95_displacement"],
        ]
        for e in results
    ]
    mae_rows = [[e["t0"], *(e["scores"][m]["mae"] for m in methods)] for e in results]
    parity_rows = [[e["t0"], *(e[p][k] for p in ("parity", "parity_padded") for k in ("mean", "max"))] for e in results]
    out = [
        "## OPERA radar: interpolating the middle frame",
        "",
        "OPERA ODYSSEY composites (2 km, 2200 x 1900, every 15 minutes), clipped to 0-60 dBZ and scaled to 0-255 "
        "for both libraries. The frame at t0 + 15 min is interpolated from t0 and t0 + 30 min with "
        "`interpolate_frames(..., t=0.5)` and flows from t0 to t0 + 30 min: Farneback (OpenCV and the port) with "
        "the livemap settings, `lucas_kanade` with its defaults and with a radar setting "
        f"({', '.join(f'{k}={v}' for k, v in LUCAS_KANADE_RADAR.items())}, chosen on other dates), a zero flow "
        "(a plain blend) and persistence of t0. *matched* is `match_distribution=True`. Scores in dBZ over the "
        "pixels covered in all three frames and above 0 dBZ in any of them or in any method's interpolation (the "
        f"same pixels for every method); CSI at {CSI_DBZ:g} dBZ. *MAE, observed rain* leaves out the pixels dry "
        "in all three frames, so it does not count rain put where none was observed.",
        "",
        "### Triplets",
        "",
        "Wet (above 0 dBZ in any frame) and scored fractions are of the covered pixels; displacement is the OpenCV "
        "Farneback flow over 30 minutes on the wet pixels.",
        "",
        table(["t0 (UTC)", "covered", "wet", "scored", "mean displacement (px)", "95th percentile (px)"], triplet_rows),
        "",
        f"### Scores, averaged over the {len(results)} triplets",
        "",
        table(
            ["flow", "MAE", "RMSE", "bias", f"CSI {CSI_DBZ:g} dBZ", "MAE, observed rain"],
            [[r, *mean[r].values()] for r in rows],
        ),
        "",
        "### MAE per triplet (dBZ, without matching)",
        "",
        table(["t0 (UTC)", *methods], mae_rows),
        "",
        "### Farneback port against OpenCV on the radar pairs (endpoint distance, pixels)",
        "",
        "Livemap settings, mean and max over the grid. With 3 levels and a 25-pixel "
        "window, the border difference of the coarsest level (displaced positions within 8 full-resolution "
        "pixels of the edge) can spread far inwards (160 pixels on 2018-08-25), so the comparison is repeated on "
        "the frames "
        f"with {PAD} zero pixels added around them (the edges of the composite are no-data, stored as 0).",
        "",
        table(["t0 (UTC)", "mean", "max", "mean, padded", "max, padded"], parity_rows),
    ]
    return "\n".join(out), {"settings": settings, "csi_dbz": CSI_DBZ, "triplets": results, "mean": mean}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="256 x 256 synthetic frames and two radar triplets")
    parser.add_argument("--json", type=Path, help="also write the results to this file")
    parser.add_argument("--opera-root", type=Path, default=Path("/disks/fast/openradar"), help="OPERA archive")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    header = (
        f"# Accuracy against OpenCV\n\ntorch {torch.__version__} on {device.type}, OpenCV {cv2.__version__}, "
        "float32 frames."
    )
    text, synthetic = run_synthetic(256 if args.quick else 512, device)
    sweep_text, sweep = run_sweep(256 if args.quick else 512, device)
    sections = [header, text, sweep_text]
    results = {
        "device": device.type,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "synthetic": synthetic,
        "sweep": sweep,
    }
    triplets = TRIPLETS[::4] if args.quick else TRIPLETS
    times = [datetime.fromisoformat(t) + timedelta(minutes=15 * k) for t in triplets for k in range(3)]
    missing = [f"{t:%Y-%m-%d %H:%M}" for t in times if not opera_path(args.opera_root, t).exists()]
    try:
        import h5py  # noqa: F401
    except ImportError:
        skip = "h5py is not installed (run with `uv run --with h5py`)"
    else:
        skip = f"no composite under {args.opera_root} for {', '.join(missing)}" if missing else None
        skip = f"{args.opera_root} not found" if not args.opera_root.is_dir() else skip
    if skip:
        log(f"radar skipped: {skip}")
        sections.append(f"## OPERA radar\n\nSkipped: {skip}.")
    else:
        text, results["radar"] = run_radar(args.opera_root, triplets, device)
        sections.append(text)
    print("\n\n".join(sections))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
