# opticalflow-torch

Differentiable optical flow in pure PyTorch: Farneback (a port of OpenCV's
`calcOpticalFlowFarneback`, with its parameters and the defaults of `FarnebackOpticalFlow.create`)
and dense Lucas-Kanade, plus warping, motion-compensated frame interpolation, flow metrics and flow
colour coding. Batched, on CPU or GPU, and differentiable with respect to the frames, the flows and
the float parameters. The only dependency is PyTorch, 2.2 or later (2.7 or later for `torch.compile`).

Work in progress: the API may still change before the first release.

## Install

Once released on PyPI:

```bash
pip install opticalflow-torch
```

Until then, from GitHub:

```bash
pip install git+https://github.com/DSIP-FBK/opticalflow-torch
```

## Example

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

# Two frames in between, at t = 1/3 and 2/3: 5-minute frames, shape (1, 2, 1, 128, 128)
frames = oft.interpolate(frame0, frame1, steps=2, data_range=60)

# As a module, here with Lucas-Kanade, whose parameters can be learned, and probability matching
of = oft.OpticalFlow("lucas_kanade", data_range=60, learnable=True)
frames = of.interpolate(frame0, frame1, steps=2, match_distribution=True)
frames.square().mean().backward()  # gradients reach of.parameters()
```

These choose between `farneback` and `lucas_kanade` (or the `Farneback` and `LucasKanade` modules)
and pass the flow to `interpolate_frames`, which takes a flow you already have:

```python
flow = oft.farneback(frame0, frame1, data_range=60)
frames = oft.interpolate_frames(frame0, frame1, flow, t=[1 / 3, 2 / 3])  # as interpolate(..., steps=2)
aligned = oft.warp(frame1, flow)  # close to frame0
```

Frames are `(B, C, H, W)` tensors and flows `(B, 2, H, W)`, with
`frame0[y, x] ≈ frame1[y + flow_y, x + flow_x]` as in OpenCV. The defaults assume frames spanning
0-255, as OpenCV's do; for other values, give their span as `data_range`, as above. With the
default, frames in [0, 1] get a Farneback flow of about zero, and Lucas-Kanade flows held back
where the texture is weak, which is easy to miss.

## Documentation

<https://dsip-fbk.github.io/opticalflow-torch/>: conventions, the Farneback port and how it
differs from OpenCV, Lucas-Kanade, interpolation, differentiability, benchmarks against OpenCV
and the API reference.

## License

Apache License 2.0, see [LICENSE](https://github.com/DSIP-FBK/opticalflow-torch/blob/main/LICENSE).
The Farneback implementation is derived from OpenCV, also under the Apache License 2.0; see
[NOTICE](https://github.com/DSIP-FBK/opticalflow-torch/blob/main/NOTICE).
