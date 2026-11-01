# Farneback

[`farneback`][opticalflow_torch.farneback.farneback] ports OpenCV's
`cv2.calcOpticalFlowFarneback` (Farnebäck, 2003)[^1] to PyTorch: same parameters, same defaults
as `cv2.FarnebackOpticalFlow.create`, same image pyramid. The
[`Farneback`][opticalflow_torch.farneback.Farneback] module wraps it with optionally learnable
parameters.

## The algorithm in brief

**Polynomial expansion.** Around each pixel, each frame is approximated by a quadratic polynomial
of the offset \(\mathbf{x}\),

\[
f(\mathbf{x}) \approx \mathbf{x}^\top A\, \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c,
\]

fitted by weighted least squares over the \((2n + 1) \times (2n + 1)\) pixels around it, with
\(n\) = `poly_n` and Gaussian weights of standard deviation `poly_sigma`. This gives
\(A_0, \mathbf{b}_0\) for `frame0` and \(A_1, \mathbf{b}_1\) for `frame1` at every pixel.

**Displacement from the coefficients.** If `frame1` is `frame0` translated by \(\mathbf{d}\), then
\(A_1 = A_0\) and \(\mathbf{b}_1 = \mathbf{b}_0 - 2 A_0 \mathbf{d}\), so
\(\mathbf{d} = -\tfrac{1}{2} A_0^{-1} (\mathbf{b}_1 - \mathbf{b}_0)\). With a current estimate
\(\tilde{\mathbf{d}}\), the coefficients of `frame1` are sampled bilinearly at
\(\mathbf{x} + \tilde{\mathbf{d}}\) and each pixel gets the equation \(A\, \mathbf{d} = \Delta\mathbf{b}\), where

\[
A = \tfrac{1}{2}\bigl(A_0(\mathbf{x}) + A_1(\mathbf{x} + \tilde{\mathbf{d}})\bigr), \qquad
\Delta\mathbf{b} = -\tfrac{1}{2}\bigl(\mathbf{b}_1(\mathbf{x} + \tilde{\mathbf{d}}) - \mathbf{b}_0(\mathbf{x})\bigr) + A\, \tilde{\mathbf{d}}.
\]

**Averaging over a window.** The normal equations \(A^\top A\, \mathbf{d} = A^\top \Delta\mathbf{b}\)
are averaged over a window around each pixel (a `winsize` box, or a Gaussian with
`window="gaussian"`), which gives a 2x2 system \(G\, \mathbf{d} = \mathbf{h}\) per pixel, solved in
closed form with `eps` added to the determinant:
\(\mathbf{d} = \operatorname{adj}(G)\, \mathbf{h} / (\det G + \text{eps})\).

**Iterations and pyramid.** The flow is re-estimated `iterations` times per level, each time from
coefficients sampled with the latest flow. Levels go from the coarsest to the full resolution;
each level's flow, resized and divided by `pyr_scale`, starts the next one. Every level is made
from the full-resolution frames: a Gaussian blur of standard deviation \((1/\text{scale} - 1)/2\)
(a 3-tap blur at full resolution), then a bilinear resize, as in OpenCV.

With `return_confidence=True` you also get the smallest eigenvalue of \(G\) at full resolution. It
is near zero where the frames are flat or show only straight edges (the aperture problem), and
large where the texture fixes both components of the flow.

## From OpenCV

`cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 25, 3, 5, 1.2, 0)` becomes

```python
farneback(a, b, pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2)
```

with `a` and `b` as `(B, C, H, W)` tensors. In full:

```python
import cv2
import torch

from opticalflow_torch import farneback

# img0, img1: (H, W) uint8 or float32 arrays
flow_cv = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 25, 3, 5, 1.2, 0)  # (H, W, 2)

frame0 = torch.from_numpy(img0).float()[None, None]  # (1, 1, H, W), values unchanged
frame1 = torch.from_numpy(img1).float()[None, None]
flow = farneback(frame0, frame1, pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2)
flow_np = flow[0].permute(1, 2, 0).numpy()  # (H, W, 2), like flow_cv
```

| `cv2.calcOpticalFlowFarneback` | `farneback` | Notes |
| --- | --- | --- |
| `prev` | `frame0` | `(B, C, H, W)` tensor instead of an `(H, W)` array |
| `next` | `frame1` | |
| `flow` | `initial_flow` | `(B, 2, H, W)`; see [the layouts](conventions.md#opencv-and-numpy-layouts) |
| `pyr_scale` | `pyr_scale` | |
| `levels` | `levels` | same value; see [below](#the-levels-parameter) |
| `winsize` | `winsize` | |
| `iterations` | `iterations` | |
| `poly_n` | `poly_n` | typically 5 or 7; OpenCV's OpenCL path accepts only these |
| `poly_sigma` | `poly_sigma` | may be a tensor, which is clamped to at least float32's epsilon instead of meaning `0.3 * poly_n` below it |
| `flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN` | `window="gaussian"` | |
| `flags=cv2.OPTFLOW_USE_INITIAL_FLOW` | pass `initial_flow` | |
| fixed to `0.3 * (winsize // 2)` | `window_sigma` | Gaussian window only |
| fixed to `1e-3` | `eps` | |
| | `return_confidence` | |

`cv2.calcOpticalFlowFarneback` has no defaults. Those of `farneback` are the defaults of
`cv2.FarnebackOpticalFlow.create`: `pyr_scale=0.5`, `levels=5`, `winsize=13`, `iterations=10`,
`poly_n=5`, `poly_sigma=1.1`, box window. Its `fastPyramids` option has no counterpart.

## The `levels` parameter

OpenCV's documentation describes `levels` as the number of pyramid layers including the initial
image, with `levels=1` meaning no extra layers. Its code adds `levels` coarser layers on top of
the full-resolution one: `levels=0` uses the full resolution only, `levels=1` adds one coarser
layer, and the two give different results. `farneback` follows the code, so the same value gives
the same pyramid as OpenCV.

As in OpenCV, coarser layers whose width or height would be below 32 pixels are dropped: on a
128 x 128 image with `pyr_scale=0.5`, `levels=5` gives the same result as `levels=2`.

## Differences from OpenCV

- **Samples that leave the image.** OpenCV samples the coefficients of `frame1` at
  \(\mathbf{x} + \tilde{\mathbf{d}}\) only when that position is inside the image, and otherwise
  switches abruptly to the coefficients of `frame0` alone, which makes the flow jump as a sample
  crosses the edge. The port turns this inside-or-outside test into a weight that goes linearly
  from 1 to 0 over the last pixel, so the flow is continuous in its inputs. Results are the same
  wherever the weight is 0 or 1, and differ from OpenCV near the image borders, where displaced
  positions fall within a pixel of the edge.
- **Several channels.** OpenCV accepts single-channel images only. Here frames may have several
  channels, whose equations are summed into one flow per image; with one channel nothing changes.
- **Precision.** OpenCV computes in float32. Here float64 frames are processed in float64, and
  float16 and bfloat16 frames in float32. Results in float32 differ from OpenCV's by round-off.
- **Additions.** Batches, GPUs, gradients, `eps` and `window_sigma` as parameters (tensors
  allowed), and the confidence output.

[^1]: G. Farnebäck, "Two-frame motion estimation based on polynomial expansion", Scandinavian
    Conference on Image Analysis (SCIA), 2003, LNCS 2749, pp. 363-370.
