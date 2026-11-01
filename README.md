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

from opticalflow_torch import farneback, warp

# A smooth random texture with values in 0-255, and a copy moved 3 pixels right and 2 up
noise = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
frame0 = 255 * F.interpolate(noise, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
motion = torch.tensor([3.0, -2.0]).view(1, 2, 1, 1).expand(1, 2, 128, 128)
frame1 = warp(frame0, -motion)

frame1.requires_grad_()
flow = farneback(frame0, frame1)  # (1, 2, 128, 128), in pixels, x first
print(flow[..., 16:-16, 16:-16].mean(dim=(-2, -1)))  # about [[3, -2]]

aligned = warp(frame1, flow)  # close to frame0
(aligned - frame0).abs().mean().backward()  # gradients reach frame1 through the flow
```

Frames are `(B, C, H, W)` tensors and flows `(B, 2, H, W)`, with
`frame0[y, x] ≈ frame1[y + flow_y, x + flow_x]` as in OpenCV. Like OpenCV, the defaults assume
values of roughly 0-255.

## Documentation

<https://dsip-fbk.github.io/opticalflow-torch/>: conventions, the Farneback port and how it
differs from OpenCV, Lucas-Kanade, interpolation, differentiability and the API reference.

## License

Apache License 2.0, see [LICENSE](https://github.com/DSIP-FBK/opticalflow-torch/blob/main/LICENSE).
The Farneback implementation is derived from OpenCV, also under the Apache License 2.0; see
[NOTICE](https://github.com/DSIP-FBK/opticalflow-torch/blob/main/NOTICE).
