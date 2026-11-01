# opticalflow-torch

Differentiable optical flow in pure PyTorch.

- [Farneback](farneback.md): a port of OpenCV's `cv2.calcOpticalFlowFarneback`, with the same
  parameters and image pyramid, and the defaults of `cv2.FarnebackOpticalFlow.create`.
- [Dense Lucas-Kanade](lucas-kanade.md), pyramidal and iterative.
- [Warping](conventions.md#warping) and resizing of flows, and
  [motion-compensated interpolation](interpolation.md) of frames, with optional probability matching.
- Endpoint error, photometric error, a soft forward-backward consistency check, and the usual
  flow colour coding for plots.

## Why

Classical optical flow is often needed inside PyTorch code: as a loss term, as an input feature,
or to align and interpolate frames. The usual implementations (OpenCV, scikit-image, pysteps) work
on NumPy arrays, one image at a time on the CPU, and cut the gradient. Here every function takes
batched tensors on the CPU or a GPU and is [differentiable](differentiability.md) with respect to
the frames, the flows and the float parameters, so gradients reach whatever produced the frames.
The only runtime dependency is PyTorch.

## Install

Once released on PyPI:

```bash
pip install opticalflow-torch
```

Until then, from GitHub:

```bash
pip install git+https://github.com/DSIP-FBK/opticalflow-torch
```

Python 3.10 or later and PyTorch 2.2 or later; `torch.compile` needs PyTorch 2.7 or later. Install
PyTorch first if you need a particular CUDA build.

## Quick start

```python
import torch
import torch.nn.functional as F

import opticalflow_torch as oft

# Two radar-like frames 15 minutes apart, in 0-60 dBZ: a smooth random field,
# then the same moved 3 pixels right and 2 up
noise = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
frame0 = 60 * F.interpolate(noise, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
motion = torch.tensor([3.0, -2.0]).view(1, 2, 1, 1).expand(1, 2, 128, 128)
frame1 = oft.warp(frame0, -motion)

flow = oft.optical_flow(frame0, frame1, data_range=60)  # Farneback; (1, 2, 128, 128), in pixels, x first
print(flow[..., 16:-16, 16:-16].mean(dim=(-2, -1)))  # about [[3, -2]]
flow = oft.optical_flow(frame0, frame1, method="lucas_kanade", data_range=60)  # or with Lucas-Kanade

# Two frames in between, at t = 1/3 and 2/3: 5-minute frames, shape (1, 2, 1, 128, 128)
frames = oft.interpolate(frame0, frame1, steps=2, data_range=60)
```

[`optical_flow`][opticalflow_torch.highlevel.optical_flow] and
[`interpolate`][opticalflow_torch.highlevel.interpolate] take the options of the chosen method,
here `data_range`; `interpolate` also takes those of the [interpolation](interpolation.md), such
as `match_distribution`. [`OpticalFlow`][opticalflow_torch.highlevel.OpticalFlow] does the same as a
module, whose parameters can be learned:

```python
of = oft.OpticalFlow("lucas_kanade", data_range=60, learnable=True)
flow = of(frame0, frame1)
frames = of.interpolate(frame0, frame1, steps=2, match_distribution=True)
frames.square().mean().backward()  # gradients reach of.parameters()
```

Underneath are [`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] (and their modules), and
[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames], which takes a flow you
already have:

```python
flow = oft.farneback(frame0, frame1, data_range=60)
frames = oft.interpolate_frames(frame0, frame1, flow, t=[1 / 3, 2 / 3])  # as interpolate(..., steps=2)
aligned = oft.warp(frame1, flow)  # close to frame0
rgb = oft.flow_to_rgb(flow)  # (1, 3, 128, 128) in [0, 1], for plotting
```

The defaults assume frames spanning 0-255, as OpenCV's do. For other values, give their span as
`data_range`, as above. With the default, frames in [0, 1] get a Farneback flow of about zero, and
Lucas-Kanade flows held back where the texture is weak, which is easy to miss.

Gradients go through the flow estimation:

```python
frame1.requires_grad_()
loss = oft.photometric_error(frame0, frame1, oft.optical_flow(frame0, frame1, data_range=60)).mean()
loss.backward()  # frame1.grad now holds d loss / d frame1
```

Read [Conventions](conventions.md) first: it fixes the direction of the flow, the meaning of
`warp`, and the [value range](conventions.md#value-range) the defaults expect.
