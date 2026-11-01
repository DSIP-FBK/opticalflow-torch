# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `farneback` and the `Farneback` module: Farneback dense optical flow ported from OpenCV's
  `calcOpticalFlowFarneback`, with OpenCV's parameter names and image pyramid, the defaults of
  `FarnebackOpticalFlow.create`, a box or Gaussian window, an initial flow, and an optional
  confidence (smallest eigenvalue per pixel).
- `lucas_kanade` and the `LucasKanade` module: dense pyramidal Lucas-Kanade with a box or Gaussian
  window, an initial flow and an optional confidence.
- Learnable `poly_sigma`, `window_sigma` and `eps` in the `Farneback` and `LucasKanade` modules.
- `warp` (bilinear, nearest or bicubic sampling; border, zeros or reflection padding) and
  `resize_flow`.
- `interpolate_frames`: motion-compensated interpolation between two frames, with an optional
  backward flow and optional probability matching.
- `endpoint_error`, `photometric_error` and `consistency` (soft forward-backward check).
- `flow_to_rgb`: the standard optical flow colour coding.
- Everything is differentiable with respect to the frames, the flows and the float parameters,
  batched, on CPU and CUDA, in float32 and float64, and works under `torch.compile` from
  PyTorch 2.7. The only runtime dependency is PyTorch, 2.2 or later.

[Unreleased]: https://github.com/DSIP-FBK/opticalflow-torch/commits/main/
