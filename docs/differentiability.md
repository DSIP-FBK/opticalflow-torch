# Differentiability

## What is differentiable

Every function is differentiable with respect to its tensor inputs, and the flow methods also
with respect to their float parameters:

| Function | Gradients reach |
| --- | --- |
| [`farneback`][opticalflow_torch.farneback.farneback], [`Farneback`][opticalflow_torch.farneback.Farneback] | `frame0`, `frame1`, `initial_flow`, `poly_sigma`, `window_sigma`, `eps` |
| [`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade], [`LucasKanade`][opticalflow_torch.lucas_kanade.LucasKanade] | `frame0`, `frame1`, `initial_flow`, `window_sigma`, `eps` |
| [`warp`][opticalflow_torch.warp.warp] | `image`, `flow` |
| [`resize_flow`][opticalflow_torch.warp.resize_flow] | `flow` |
| [`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] | `frame0`, `frame1`, `flow`, `backward_flow`, `t` (as a tensor) |
| [`optical_flow`][opticalflow_torch.highlevel.optical_flow], [`interpolate`][opticalflow_torch.highlevel.interpolate], [`OpticalFlow`][opticalflow_torch.highlevel.OpticalFlow] | as the method and the interpolation they call |
| [`endpoint_error`][opticalflow_torch.metrics.endpoint_error], [`photometric_error`][opticalflow_torch.metrics.photometric_error], [`consistency`][opticalflow_torch.metrics.consistency], [`flow_to_rgb`][opticalflow_torch.visualize.flow_to_rgb] | their tensor inputs |

Parameters typed `float | Tensor` accept tensors that require grad. Integer, string and boolean
arguments (`levels`, `winsize`, `iterations`, `poly_n`, `window`, `quantiles`, ...),
`pyr_scale`, which sets the pyramid sizes, and `data_range` choose the computation and have no
gradient.

There are no hard thresholds, argmax or data-dependent branches. The non-smooth points are kinks
like those of bilinear interpolation: the gradient with respect to a sampling position changes
when the position crosses a pixel, and with `padding_mode="border"` it is zero for positions
beyond the edge. The probability matching of `interpolate_frames` has kinks of the same kind where
values cross quantiles, and stays continuous where they tie (see
[Probability matching](interpolation.md#probability-matching)). A `poly_sigma` below float32's
epsilon selects OpenCV's default when it is a number, and is clamped when it is a tensor.

## Learnable parameters

[`Farneback`][opticalflow_torch.farneback.Farneback] and
[`LucasKanade`][opticalflow_torch.lucas_kanade.LucasKanade] are `torch.nn.Module`s. With
`learnable=True` they register `poly_sigma` (Farneback only), `window_sigma` (Gaussian window
only) and `eps` as parameters; `learnable` also takes a list of names, e.g. `learnable=["eps"]`.

```python
import torch

from opticalflow_torch import Farneback, endpoint_error

model = Farneback(window="gaussian", learnable=True)  # poly_sigma, window_sigma and eps
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for frame0, frame1, reference in loader:  # frames and reference flows
    loss = endpoint_error(model(frame0, frame1), reference).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The parameters are stored as logarithms, so they stay positive and an optimiser step changes them
by a factor rather than an amount; the properties `poly_sigma`, `window_sigma` and `eps` give
their current values. [`OpticalFlow`][opticalflow_torch.highlevel.OpticalFlow] holds one of them
as `estimator`, e.g. `of.estimator.eps`. Without reference flows, minimise
[`photometric_error`][opticalflow_torch.metrics.photometric_error] instead.

## Flow inside a loss

Because gradients reach the frames, a flow can be part of a training loss. For instance, to make
a forecasting model move its fields like the observations do:

```python
import torch

from opticalflow_torch import endpoint_error, farneback


def motion(frames: torch.Tensor) -> torch.Tensor:
    """Flows between consecutive frames of (B, T, H, W) sequences, shape (B * (T - 1), 2, H, W)."""
    frame0, frame1 = frames[:, :-1].flatten(0, 1).unsqueeze(1), frames[:, 1:].flatten(0, 1).unsqueeze(1)
    return farneback(frame0, frame1, data_range=1.0)  # fields in [0, 1], see Conventions


with torch.no_grad():
    target = motion(observed)
loss = endpoint_error(motion(forecast), target).mean()  # gradients reach the forecast
```

## Pitfalls

**Flat regions and `eps`.** Where a window has no texture (no rain, a uniform background, or only
a straight edge), the 2x2 system of the pixel is close to singular and the flow is set by `eps`
rather than by the frames: Farneback's flow goes to zero there, Lucas-Kanade's comes from the
coarser levels and the neighbouring pixels. `eps` decides how much contrast counts as texture,
relative to `data_range`, so set `data_range` to the span of the values (see
[Value range](conventions.md#value-range)). To discount such pixels in a loss, weight them by the
confidence output (`return_confidence=True`).

**Memory of the unrolled iterations.** Autograd keeps the intermediate tensors of every iteration
at every pyramid level, so the memory of the backward pass grows with `iterations` and with the
batch and image size. Use fewer iterations for training,
`torch.utils.checkpoint.checkpoint(farneback, frame0, frame1, use_reentrant=False)` to recompute
the forward pass during the backward pass instead of storing it, and `torch.no_grad()` for flows
you do not differentiate.

**Round-off of `warp` in float32.** `torch.nn.functional.grid_sample` takes positions scaled to
[-1, 1], so pixel positions are converted back and forth, which costs up to about machine epsilon
times the image size: around 1e-5 pixels for 512 pixels and 6e-5 pixels for 2048 pixels in
float32. A zero flow therefore does not return the image bit for bit, and finite-difference
gradient checks need float64. Use float64 where this matters.

**Non-finite flows.** Where a flow is NaN or infinite, [`warp`][opticalflow_torch.warp.warp]
returns NaN, with zero gradient. The gradients of the other pixels are unaffected, but a loss
summed over all pixels is NaN: reduce it with `torch.nansum` or `torch.nanmean`, or select the
finite pixels first. NaNs in the frames spread through the filters of the flow methods, so fill
missing values before estimating a flow. In
[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] with
`match_distribution=True`, a NaN or infinity in the frames or flows makes most of its image NaN, as
the distributions are per image.

**Half precision.** [`warp`][opticalflow_torch.warp.warp],
[`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] compute float16 and bfloat16 inputs
in float32 and return the input dtype: half precision cannot even represent sub-pixel positions on
large images. Inside `torch.autocast` regions the functions and modules compute as outside them, so
a mixed precision training step (autocast with a `torch.amp.GradScaler`, `torch.cuda.amp.GradScaler`
before PyTorch 2.3) can include them. They do not promote mixed dtypes as autocast does, though: the
frames and flows of a call must share one dtype, and a model's float16 output gives a float16 flow.
Cast it first, as in `endpoint_error(motion(forecast.float()), target)` for the example above.
