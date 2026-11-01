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

from opticalflow_torch import farneback, flow_to_rgb, photometric_error, warp

# A smooth random texture with values in 0-255, and a copy moved 3 pixels right and 2 up
noise = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
frame0 = 255 * F.interpolate(noise, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
motion = torch.tensor([3.0, -2.0]).view(1, 2, 1, 1).expand(1, 2, 128, 128)
frame1 = warp(frame0, -motion)

flow = farneback(frame0, frame1)  # (1, 2, 128, 128), in pixels, x first
print(flow[..., 16:-16, 16:-16].mean(dim=(-2, -1)))  # about [[3, -2]]

aligned = warp(frame1, flow)  # close to frame0
rgb = flow_to_rgb(flow)  # (1, 3, 128, 128) in [0, 1], for plotting
```

The defaults assume frames spanning 0-255, as OpenCV's do. For other values, give their span as
`data_range`. With the default, frames in [0, 1] get a Farneback flow of about zero, and
Lucas-Kanade flows held back where the texture is weak, which is easy to miss:

```python
flow = farneback(frame0 / 255, frame1 / 255, data_range=1.0)  # frames in [0, 1], the same flow
```

Gradients go through the flow estimation:

```python
frame1.requires_grad_()
loss = photometric_error(frame0, frame1, farneback(frame0, frame1)).mean()
loss.backward()  # frame1.grad now holds d loss / d frame1
```

Read [Conventions](conventions.md) first: it fixes the direction of the flow, the meaning of
`warp`, and the [value range](conventions.md#value-range) the defaults expect.
