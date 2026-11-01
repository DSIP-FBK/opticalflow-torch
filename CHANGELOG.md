# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `optical_flow`, `interpolate` and the `OpticalFlow` module: one entry point for both methods,
  chosen with `method="farneback"` or `"lucas_kanade"`; `interpolate` computes the flow and the
  frames in between, e.g. `steps=2` for two frames at times 1/3 and 2/3.
- `farneback` and the `Farneback` module: Farneback dense optical flow ported from OpenCV's
  `calcOpticalFlowFarneback`, with OpenCV's parameter names and image pyramid, the defaults of
  `FarnebackOpticalFlow.create`, a box or Gaussian window, an initial flow, and an optional
  confidence (smallest eigenvalue per pixel).
- `lucas_kanade` and the `LucasKanade` module: dense pyramidal Lucas-Kanade with a box or Gaussian
  window, an initial flow and an optional confidence.
- Learnable `poly_sigma`, `window_sigma` and `eps` in the `Farneback` and `LucasKanade` modules.
- `data_range` in `farneback`, `Farneback`, `lucas_kanade` and `LucasKanade`: the span of the frame
  values, 255 by default. `eps` is given for frames spanning 0-255 and acts as
  `eps * (data_range / 255) ** 4` (Farneback) or `** 2` (Lucas-Kanade) on others, so frames
  scaled together with `data_range` give the same flow; e.g. `data_range=1.0` for frames in
  [0, 1], which otherwise get a Farneback flow of about zero and Lucas-Kanade flows held back
  where the texture is weak.
- `warp` (bilinear, nearest or bicubic sampling; border, zeros or reflection padding) and
  `resize_flow`.
- `interpolate_frames`: motion-compensated interpolation between two frames, with an optional
  backward flow and optional probability matching.
- `endpoint_error`, `photometric_error` and `consistency` (soft forward-backward check).
- `flow_to_rgb`: the standard optical flow colour coding.
- Everything is differentiable with respect to the frames, the flows and the float parameters,
  batched, on CPU and CUDA, in float32 and float64, and works under `torch.compile` from
  PyTorch 2.7. The only runtime dependency is PyTorch, 2.2 or later.
- Every function and module gives the same results inside and outside `torch.autocast`: autocast
  is turned off while it runs, so mixed precision training does not push its convolutions to
  float16 or bfloat16, where squared gradients of 0-255 frames overflow.

[Unreleased]: https://github.com/DSIP-FBK/opticalflow-torch/commits/main/
