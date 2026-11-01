# Benchmarks

Run times and accuracy against OpenCV, from the two scripts in
[`benchmarks/`](https://github.com/DSIP-FBK/opticalflow-torch/tree/main/benchmarks):

- `speed.py`: time per pair of frames, CUDA memory and `torch.compile` time.
- `accuracy.py`: errors on synthetic pairs with exact flows, the distance between the Farneback
  port and OpenCV, and frame interpolation on OPERA radar composites.

[`farneback`][opticalflow_torch.farneback.farneback] is compared with
`cv2.calcOpticalFlowFarneback`, which it ports, so the two should agree; this is a correctness
check, not a design goal. The opencv-python wheels have no dense Lucas-Kanade
(`cv::cuda::DensePyrLKOpticalFlow` needs a CUDA build of OpenCV with the contrib modules), only
the sparse `cv2.calcOpticalFlowPyrLK`, a different algorithm (see
[Lucas-Kanade](lucas-kanade.md#compared-with-opencvs-calcopticalflowpyrlk)): it is timed with
every pixel as a point and compared for accuracy at grid points.

## Setup

| | |
| --- | --- |
| CPU | AMD Ryzen 9 5950X, 16 cores, 32 threads |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Software | Python 3.12.3, PyTorch 2.15.0+cu132 (16 CPU threads), opencv-python-headless 5.0.0 (CPU with 32 threads, and OpenCL on the RTX 5090; no CUDA), NumPy 2.5.4 |
| Code | commit `4b5ee24`, 2026-11-01 |

Both libraries get the same float32 frames in 0-255 (uint8 for `calcOpticalFlowPyrLK`, which
accepts nothing else). Two Farneback settings are used: the OpenCV defaults (`pyr_scale=0.5`,
`levels=5`, `winsize=13`, `iterations=10`, `poly_n=5`, `poly_sigma=1.1`) and the settings used
for radar, "livemap" (`levels=3`, `winsize=25`, `iterations=3`, `poly_sigma=1.2`, the rest the
same). Box window in both libraries.

OpenCV runs on the GPU through OpenCL when given `cv2.UMat` images: the same functions, from the
same PyPI wheel. In a separate check, its OpenCL Farneback flows agreed with its CPU flows to
2.5e-5 pixels beyond 32 pixels from the edges.

## Speed

Frames: a smooth random texture and the same texture shifted by (2.5, -1.5) pixels, at three
square sizes and on the OPERA grid (2200 x 1900). Inputs are on the device before timing (and
uploaded as `cv2.UMat` for OpenCL); outputs stay there. Each cell is the median of at least 5
runs (about 1 s of runs) after warm-up, CUDA and OpenCL synchronised (`cv2.ocl.finish()`). B is
the number of pairs per call, or of calls, and times are per pair. The interquartile range is
within 5% of the median in most cells and at most 27% (OpenCV on the CPU at 256 x 256 and
512 x 512); `--json` saves every run.

### Farneback, time per pair (ms)

| OpenCV defaults | B | 256 x 256 | 512 x 512 | 1024 x 1024 | 2200 x 1900 |
| --- | --- | --- | --- | --- | --- |
| `cv2`, CPU | 1 | 19.9 | 82.8 | 399 | 1430 |
| `cv2`, CPU, `setNumThreads(1)` | 1 | 19.9 | 82.5 | 397 | 1523 |
| `cv2`, OpenCL | 1 | 3.15 | 3.84 | 5.94 | 12.1 |
| `farneback`, CPU, 16 threads | 1 | 64.9 | 171 | 1098 | 8012 |
| `farneback`, CUDA | 1 | 39.4 | 49.9 | 61.4 | 99.0 |
| `farneback`, CUDA, `torch.compile` | 1 | 3.69 | 5.19 | 11.6 | 49.6 |
| `farneback`, CUDA, forward + backward | 1 | 117 | 150 | 184 | 272 |
| `cv2`, CPU, loop of 8 calls | 8 | 19.9 | 83.7 | 404 | 1466 |
| `cv2`, CPU, 16 calls in 16 Python threads | 16 | 2.04 | 10.5 | 49.6 | 233 |
| `cv2`, OpenCL, loop of 8 calls | 8 | 2.62 | 4.05 | 5.89 | 12.1 |
| `farneback`, CPU, 16 threads | 8 | 57.1 | 457 | | |
| `farneback`, CUDA | 8 | 5.01 | 7.58 | 21.4 | 86.4 |

| livemap | B | 256 x 256 | 512 x 512 | 1024 x 1024 | 2200 x 1900 |
| --- | --- | --- | --- | --- | --- |
| `cv2`, CPU | 1 | 8.27 | 36.4 | 137 | 641 |
| `cv2`, CPU, `setNumThreads(1)` | 1 | 8.28 | 33.0 | 136 | 634 |
| `cv2`, OpenCL | 1 | 1.67 | 2.45 | 4.00 | 7.53 |
| `farneback`, CPU, 16 threads | 1 | 25.6 | 63.9 | 523 | 3143 |
| `farneback`, CUDA | 1 | 15.8 | 16.0 | 16.9 | 29.6 |
| `farneback`, CUDA, `torch.compile` | 1 | 1.65 | 2.02 | 4.83 | 21.0 |
| `farneback`, CUDA, forward + backward | 1 | 41.3 | 42.5 | 42.8 | 69.2 |
| `cv2`, CPU, loop of 8 calls | 8 | 8.25 | 37.3 | 159 | 669 |
| `cv2`, CPU, 16 calls in 16 Python threads | 16 | 0.98 | 4.54 | 23.4 | 113 |
| `cv2`, OpenCL, loop of 8 calls | 8 | 1.49 | 2.61 | 3.96 | 7.81 |
| `farneback`, CPU, 16 threads | 8 | 17.0 | 168 | | |
| `farneback`, CUDA | 8 | 2.02 | 2.53 | 6.81 | 29.0 |

`cv2`, CPU is a plain call; OpenCV may use 32 threads, but `calcOpticalFlowFarneback` runs
almost entirely on one (`setNumThreads(1)` is at most 7% slower). Parallel calls in Python
threads, one per physical core, are how OpenCV uses the whole CPU. Forward + backward is the
flow and its gradient with respect to `frame0`. The CPU is not timed at B=8 beyond 512 x 512,
which would take minutes.

### Lucas-Kanade, time per pair (ms)

| | 256 x 256 | 512 x 512 | 1024 x 1024 | 2200 x 1900 |
| --- | --- | --- | --- | --- |
| `cv2.calcOpticalFlowPyrLK`, every pixel, CPU, 32 threads | 38.9 | 157 | 624 | 2499 |
| `cv2.calcOpticalFlowPyrLK`, every pixel, OpenCL | 1.64 | 4.25 | 14.3 | 56.1 |
| `lucas_kanade`, CPU, 16 threads | 37.6 | 83.7 | 651 | 3900 |
| `lucas_kanade`, CUDA | 25.7 | 25.6 | 25.9 | 38.7 |
| `lucas_kanade`, CUDA, forward + backward | 70.1 | 70.5 | 71.4 | 99.5 |

`calcOpticalFlowPyrLK` with `winSize=(21, 21)` and `maxLevel=3` (its defaults), at most 5
iterations per level to match `iterations=5` of `lucas_kanade` (OpenCV's default is 30, about
13% slower here), and OpenCV's early stop at updates below 0.01 pixels. Unlike Farneback, it
does use all 32 threads (one thread is about 14x slower). The texture covers every pixel, so it
does its full work; on radar frames it would stop early in dry areas, while `lucas_kanade` costs
the same whatever the content.

### CUDA memory and compilation

Peak CUDA memory of one call (MiB, inputs included):

| | B | 256 x 256 | 512 x 512 | 1024 x 1024 | 2200 x 1900 |
| --- | --- | --- | --- | --- | --- |
| `farneback`, either setting | 1 | 17 | 70 | 276 | 1103 |
| `farneback`, either setting, `torch.compile` | 1 | 8 | 33 | 128 | 512 |
| `farneback`, either setting | 8 | 137 | 547 | 2188 | 8729 |
| `farneback`, OpenCV defaults, forward + backward | 1 | 305 | 1242 | 4895 | 19467 |
| `farneback`, livemap, forward + backward | 1 | 105 | 421 | 1658 | 6589 |
| `lucas_kanade` | 1 | 12 | 48 | 188 | 752 |
| `lucas_kanade`, forward + backward | 1 | 137 | 542 | 2153 | 8560 |

First call of `torch.compile(farneback, fullgraph=True)`, with cold caches (s):

| | 256 x 256 | 512 x 512 | 1024 x 1024 | 2200 x 1900 |
| --- | --- | --- | --- | --- |
| OpenCV defaults | 58.7 | 73.4 | 89.5 | 91.6 |
| livemap | 26.2 | 25.0 | 25.2 | 26.2 |

Compilation takes longer with more pyramid levels and iterations, which are unrolled; levels
under 32 pixels are skipped, so small frames have fewer. OpenCV's first OpenCL call also builds
its kernels; the warm-up leaves it out, and OpenCV caches the built kernels on disk.

## Accuracy

### Synthetic pairs

512 x 512 frames from an analytic texture (values 0-255, wavelengths 8 to 64 pixels) with exact
flows: translations, rotations and a smooth non-rigid drift, scaled so that the largest interior
displacement is 1.5 ("small"), 5 ("medium") or 18 pixels ("large"). The noise case adds Gaussian
noise of standard deviation 4 to each frame. Scores exclude a 32-pixel border band.

Mean interior endpoint error (pixels). OpenCV and the port give the same numbers to the digits
shown, so each Farneback column stands for both:

| case | mean / max displacement | Farneback, defaults | Farneback, livemap | `lucas_kanade` |
| --- | --- | --- | --- | --- |
| translation, small | 1.5 / 1.5 | 0.011 | 0.0081 | 0.024 |
| rotation, small | 0.8 / 1.5 | 0.014 | 0.0091 | 0.023 |
| non-rigid, small | 1.0 / 1.5 | 0.015 | 0.013 | 0.025 |
| translation, medium | 5.0 / 5.0 | 0.011 | 0.0081 | 0.025 |
| rotation, medium | 2.7 / 5.0 | 0.034 | 0.024 | 0.024 |
| non-rigid, medium | 3.4 / 5.0 | 0.037 | 0.038 | 0.025 |
| translation, large | 18.0 / 18.0 | 0.0092 | 0.0072 | 9.311 |
| rotation, large | 9.8 / 18.0 | 0.117 | 0.086 | 0.057 |
| non-rigid, large | 12.2 / 18.0 | 0.129 | 0.138 | 0.025 |
| non-rigid, medium, noise | 3.4 / 5.0 | 0.047 | 0.042 | 0.052 |

Distance between the flows of `farneback` and `cv2.calcOpticalFlowFarneback` with the same
settings (pixels): mean and maximum over the image, and maximum beyond 32 and beyond 96 pixels
from the edges. The large values near the edges are the documented
[border difference](farneback.md#differences-from-opencv); at the coarse pyramid levels the
window spreads it inwards, by up to 35 pixels on the large livemap cases.

| case | defaults, mean | defaults, max | defaults, max >32 px | defaults, max >96 px | livemap, mean | livemap, max | livemap, max >32 px | livemap, max >96 px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| translation, small | 3.2e-4 | 0.110 | 5.1e-6 | 3.0e-6 | 1.8e-4 | 0.029 | 1.4e-6 | 1.1e-6 |
| rotation, small | 3.3e-4 | 0.242 | 4.6e-6 | 4.6e-6 | 2.3e-4 | 0.060 | 1.7e-6 | 7.4e-7 |
| non-rigid, small | 3.3e-4 | 0.198 | 3.9e-6 | 3.9e-6 | 2.4e-4 | 0.058 | 1.5e-6 | 8.3e-7 |
| translation, medium | 2.6e-3 | 1.716 | 6.0e-6 | 6.0e-6 | 4.6e-4 | 0.057 | 3.4e-6 | 2.9e-6 |
| rotation, medium | 5.4e-4 | 0.216 | 4.9e-6 | 4.9e-6 | 2.4e-4 | 0.041 | 2.7e-6 | 1.7e-6 |
| non-rigid, medium | 8.5e-4 | 0.301 | 4.1e-6 | 4.1e-6 | 3.2e-4 | 0.036 | 2.5e-6 | 2.4e-6 |
| translation, large | 9.1e-3 | 1.767 | 2.5e-5 | 1.2e-5 | 2.2e-3 | 0.266 | 4.5e-4 | 9.1e-6 |
| rotation, large | 5.0e-3 | 0.674 | 8.5e-6 | 6.8e-6 | 2.2e-3 | 0.169 | 3.6e-4 | 5.2e-6 |
| non-rigid, large | 9.2e-3 | 1.896 | 1.1e-5 | 8.8e-6 | 3.5e-3 | 0.219 | 1.1e-3 | 7.7e-6 |
| non-rigid, medium, noise | 8.4e-4 | 0.293 | 4.5e-6 | 3.5e-6 | 3.2e-4 | 0.036 | 2.5e-6 | 2.2e-6 |

Against the sparse `cv2.calcOpticalFlowPyrLK` (same settings as in the timing), at points every
16 pixels in the interior (841 points, none lost by OpenCV), on the frames rounded to uint8;
`lucas_kanade` with its defaults on the same frames, read at the same points. Mean endpoint
error (pixels):

| case | `calcOpticalFlowPyrLK` | `lucas_kanade` |
| --- | --- | --- |
| translation, small | 0.0086 | 0.025 |
| rotation, small | 0.012 | 0.024 |
| non-rigid, small | 0.014 | 0.026 |
| translation, medium | 0.0086 | 0.025 |
| rotation, medium | 0.033 | 0.024 |
| non-rigid, medium | 0.041 | 0.025 |
| translation, large | 25.283 | 9.507 |
| rotation, large | 0.110 | 0.108 |
| non-rigid, large | 0.142 | 0.026 |
| non-rigid, medium, noise | 0.046 | 0.053 |

### Uniform motions of growing length

How large a motion a pyramid method follows depends on the texture as well as on the method.
Translations at 33 degrees on 512 x 512 frames of the same texture with wavelengths up to 64
pixels (as above) and up to 256 pixels; mean endpoint error at the grid points above (pixels).
`calcOpticalFlowPyrLK` on uint8 frames, the others on float32 frames.

| longest wavelength | shift (px) | Farneback, defaults | Farneback, livemap | `lucas_kanade` | `lucas_kanade`, `levels=5` | `calcOpticalFlowPyrLK` |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | 0.012 | 0.0071 | 0.029 | 0.029 | 0.0085 |
| 64 | 16 | 0.012 | 0.0062 | 0.030 | 0.290 | 0.0080 |
| 64 | 24 | 0.0073 | 27.984 | 29.584 | 2.332 | 28.714 |
| 64 | 32 | 0.019 | 28.143 | 30.326 | 7.778 | 28.731 |
| 64 | 48 | 0.051 | 28.585 | 42.878 | 27.651 | 49.598 |
| 256 | 8 | 0.011 | 0.0070 | 0.027 | 0.027 | 0.0085 |
| 256 | 16 | 0.011 | 0.0063 | 0.028 | 0.028 | 0.0083 |
| 256 | 24 | 0.0077 | 0.010 | 0.013 | 0.013 | 0.029 |
| 256 | 32 | 0.025 | 0.015 | 1.776 | 0.026 | 2.905 |
| 256 | 48 | 85.045 | 59.759 | 57.309 | 56.684 | 58.246 |

### OPERA radar: interpolating the middle frame

OPERA ODYSSEY composites (2 km, 2200 x 1900, every 15 minutes), clipped to 0-60 dBZ and scaled
to 0-255. For eight triplets (t0, t0 + 15 min, t0 + 30 min), two per season from different
years, the middle frame is interpolated from the outer two with
[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] (`t=0.5`) and scored
against the real one, in dBZ. Flows: Farneback with the livemap settings (OpenCV and the port);
`lucas_kanade` with its defaults and with a radar setting, `levels=4, winsize=51`, the best mean
MAE among levels 3-5 and windows of 21-101 pixels on five other triplets (2017, 2019, 2022,
2024); a zero flow (a plain blend of the two frames); and persistence of the first frame.
"Matched" is `match_distribution=True`.

Scores are over the pixels covered by radar in all three frames and above 0 dBZ in any of them
or in any method's interpolation, the same pixels for every method, so rain put where none was
observed counts. The last column leaves those pixels out (above 0 dBZ in an observed frame
only), so it does not count that rain.

| flow | MAE | RMSE | bias | CSI at 20 dBZ | MAE, observed rain only |
| --- | --- | --- | --- | --- | --- |
| Farneback (OpenCV = port) | 3.091 | 4.924 | -0.015 | 0.713 | 3.309 |
| Farneback (OpenCV = port), matched | 3.054 | 5.053 | -0.019 | 0.722 | 3.349 |
| `lucas_kanade` | 3.270 | 5.284 | -0.048 | 0.694 | 3.543 |
| `lucas_kanade`, matched | 3.257 | 5.419 | -0.014 | 0.701 | 3.594 |
| `lucas_kanade`, radar setting | 3.135 | 5.008 | -0.020 | 0.707 | 3.369 |
| `lucas_kanade`, radar setting, matched | 3.108 | 5.142 | -0.014 | 0.715 | 3.419 |
| zero flow | 4.854 | 7.234 | 0.006 | 0.552 | 5.564 |
| zero flow, matched | 4.932 | 7.591 | -0.023 | 0.572 | 5.653 |
| persistence | 5.951 | 9.319 | -0.071 | 0.491 | 6.824 |

Per triplet: wet fraction of the covered pixels (above 0 dBZ in an observed frame) and scored
fraction, displacement of the wet pixels over 30 minutes (OpenCV Farneback; mean and 95th
percentile, pixels), MAE without matching (dBZ), and the distance between the port's and
OpenCV's flows (pixels, maximum over the grid).

| t0 (UTC) | wet | scored | displacement | MAE Farneback | MAE `lucas_kanade` | MAE `lucas_kanade`, radar | MAE zero flow | port vs OpenCV | same, padded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014-02-01 12:00 | 33.1% | 37.9% | 10.6 / 20.9 | 2.635 | 2.784 | 2.693 | 3.890 | 4.7e-4 | 4.4e-4 |
| 2020-02-16 13:00 | 37.3% | 40.9% | 15.4 / 27.0 | 2.920 | 3.509 | 3.123 | 4.602 | 5.7e-4 | 6.8e-4 |
| 2015-05-19 12:00 | 33.8% | 39.9% | 10.5 / 17.3 | 3.244 | 3.329 | 3.256 | 5.105 | 2.9e-4 | 6.0e-4 |
| 2021-05-16 12:00 | 29.3% | 34.0% | 9.0 / 19.0 | 3.372 | 3.472 | 3.373 | 5.654 | 3.2e-4 | 4.1e-4 |
| 2016-06-23 14:00 | 15.1% | 17.6% | 12.7 / 23.1 | 3.225 | 3.260 | 3.228 | 4.769 | 5.9e-4 | 7.6e-4 |
| 2018-08-25 13:00 | 27.8% | 31.9% | 11.2 / 24.7 | 3.559 | 3.671 | 3.557 | 5.779 | 1.361 | 4.6e-4 |
| 2013-11-04 13:00 | 34.3% | 39.6% | 12.0 / 22.9 | 2.956 | 3.092 | 2.979 | 4.355 | 4.4e-4 | 1.9e-3 |
| 2023-10-19 12:00 | 36.8% | 41.3% | 13.2 / 22.2 | 2.819 | 3.039 | 2.872 | 4.679 | 4.1e-4 | 3.9e-4 |

On 2018-08-25, rain moves north near the top edge, and at the coarsest pyramid level the
displaced positions reach the edge, where the port differs from OpenCV by design; the 25-pixel
window spreads this up to 160 pixels inwards. With 128 zero pixels added around both frames
("padded"; the edges of the composite are no-data, stored as 0), the maximum is 4.6e-4 pixels.
The mean distance is about 1e-6 pixels on every triplet once padded (3e-4 on 2018-08-25 without
padding).

## Reading the results

- **Parity.** Away from the edges the port matches OpenCV to round-off: at most 2.5e-5 pixels
  beyond 32 pixels from the edges with the defaults, and at most 1.2e-5 with either setting
  beyond 96 pixels. With livemap, the border difference reaches past the 32-pixel band on the
  large motions (up to 1.1e-3 pixels at 35 pixels from the edge). On the radar grid the largest
  difference is 4e-4 to 2e-3 pixels on every triplet once padded, against a mean of about 1e-6.
  Endpoint errors and radar scores are the same to the digits shown.
- **GPU.** OpenCV's own GPU path, OpenCL through `cv2.UMat`, is the fastest way to get one
  Farneback flow on this machine: 12 ms (defaults) and 7.5 ms (livemap) on the OPERA grid, 120x
  and 85x faster than OpenCV on the CPU. `farneback` on CUDA takes 99 and 30 ms (14x and 22x
  faster than OpenCV on the CPU, 8x and 4x slower than OpenCL), and 50 and 21 ms with
  `torch.compile` (4x and 3x slower than OpenCL). Only with livemap on frames of 512 x 512 or
  smaller does the compiled port match or beat OpenCL (2.0 against 2.5 ms at 512 x 512). Batches
  do not change this on the OPERA grid (86 and 29 ms per pair at B=8). The port's case is
  elsewhere: gradients, batches inside PyTorch code, and frames that never leave the GPU.
- **Small frames.** Without `torch.compile`, CUDA calls are bound by kernel launches: a livemap
  call launches about 2200 kernels, about 16 ms in all, while the GPU computes for 3 ms at
  512 x 512 and 21 ms on the OPERA grid. Batches or `torch.compile` remove the launch cost.
- **CPU.** One `farneback` call on 16 cores is 2-6x slower than one OpenCV call, which runs
  almost entirely on one core. With the whole CPU, 16 OpenCV calls in parallel threads are 6-10x
  faster per pair than one call, and much faster than `farneback` on the CPU. Batches of 8 make
  `farneback` 12-34% faster per pair at 256 x 256 and 2.6-2.7x slower at 512 x 512.
- **Lucas-Kanade.** On the OPERA grid, the dense `lucas_kanade` on CUDA computes a flow at every
  pixel 65x faster than `calcOpticalFlowPyrLK` tracks every pixel on the CPU, and 1.4x faster
  than it does with OpenCL; on 1024 x 1024 frames and below, OpenCL PyrLK is faster (1.6 to
  14 ms against 26). The error of `lucas_kanade` stays at 0.025-0.06 pixels as rotations and non-rigid flows
  grow, where Farneback's rises (0.09-0.14 pixels on the large ones); on small motions and with
  noise Farneback is more accurate. How large a motion it follows depends on the texture: with
  wavelengths up to 64 pixels, its defaults (3 levels) lose translations beyond 16 pixels
  (9.5 pixels of error at the grid points on the 18-pixel translation, where PyrLK is off by
  25), while Farneback livemap, also with 3 levels, holds up to 18 pixels and loses 24; with
  wavelengths up to 256 pixels, `lucas_kanade` holds up to 24 pixels. PyrLK, with the same 3 levels, fails
  at about the same shifts.
- **Radar.** Interpolation with Farneback flows about halves the MAE of persistence, and is
  better than with `lucas_kanade` (defaults) on every triplet. Much of the gap is in the
  settings: with the radar setting chosen on other dates, `lucas_kanade` comes within 0.05 dBZ
  of Farneback on average, ties on two triplets and is furthest behind on the fastest motion
  (2020-02-16). With the flows, `match_distribution=True` lowers MAE a little and raises CSI, at
  the cost of RMSE. Its MAE gain comes from the pixels dry in all three frames, where unmatched
  blends spread light rain: without them, the MAE rises instead.
- **What OpenCV does not offer.** Gradients through the flow (forward + backward costs 2.3-3x
  the forward pass), batches in one call, and CUDA tensors in and out: OpenCV's own CUDA
  Farneback needs a custom build and was not measured, and its OpenCL path takes `cv2.UMat`
  images, not tensors.

## Rerun

```bash
uv run python benchmarks/speed.py --json /tmp/speed.json > /tmp/speed.md   # about 15 minutes
uv run python benchmarks/speed.py --quick                                  # smoke test
uv run --with h5py python benchmarks/accuracy.py --json /tmp/accuracy.json > /tmp/accuracy.md
```

`speed.py` takes `--no-compile` and `--cpu-only` (no CUDA and no OpenCL rows). `accuracy.py`
takes `--quick`, `--device` and `--opera-root` (the root of the OPERA archive,
`/disks/fast/openradar` by default); its radar part is skipped without h5py or the archive.
