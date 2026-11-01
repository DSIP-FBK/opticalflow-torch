# Conventions

## Shapes

Frames are floating-point tensors of shape `(B, C, H, W)`. The images of a batch are independent.
`frame0` and `frame1` must have the same shape, dtype and device.

Flows are tensors of shape `(B, 2, H, W)`, in pixels, with the dtype and device of the frames:

| Channel | Component | Axis | Positive towards |
| --- | --- | --- | --- |
| `flow[:, 0]` | x, written \(u\) | columns | the right |
| `flow[:, 1]` | y, written \(v\) | rows | the bottom |

Frames with several channels give one flow per image: the flow methods sum the equations of the
channels. [`warp`][opticalflow_torch.warp.warp] and
[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] treat each channel
separately with the same flow.

Integer coordinates are pixel centres, as in OpenCV.

## Direction of the flow

A flow from `frame0` to `frame1` satisfies

\[
\text{frame0}[y, x] \approx \text{frame1}[y + v(y, x),\; x + u(y, x)],
\]

which is OpenCV's convention for `cv2.calcOpticalFlowFarneback(prev, next, ...)`. The flow lives on
the grid of `frame0` and points to where the content of each pixel has gone in `frame1`: content
moving right has \(u > 0\). For consecutive radar images this is the forward motion, in pixels per
time step.

## Warping

[`warp`][opticalflow_torch.warp.warp] samples an image at the positions displaced by a flow:

\[
\text{warp(image, flow)}[y, x] = \text{image}[y + v(y, x),\; x + u(y, x)],
\]

with bilinear interpolation by default. Hence

```python
warp(frame1, flow)  # ≈ frame0
```

This is backward warping: each output pixel fetches its value, so there are no holes. To move
`frame0` towards `frame1` you need the flow from `frame1` to `frame0` (`warp(frame0, backward_flow)`
≈ `frame1`), or, when the flow varies slowly, `warp(frame0, -flow)`.
[`interpolate_frames`][opticalflow_torch.interpolate.interpolate_frames] does this for fractions of
the motion.

Positions outside the image take the value given by `padding_mode`: `"border"` (the default)
repeats the edge pixels, `"zeros"` uses zeros and `"reflection"` mirrors the image about the
centres of its edge pixels. With the defaults, `warp(image, flow)` computes what
`cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)` computes with
`map_x` \(= x + u\) and `map_y` \(= y + v\), up to rounding: OpenCV 4 rounds the positions to
1/32 pixel, OpenCV 5 does not.

[`resize_flow`][opticalflow_torch.warp.resize_flow] resizes a flow and rescales its vectors to the
new pixel size.

## OpenCV and NumPy layouts

OpenCV stores a flow as an `(H, W, 2)` array, with the same channel order and direction:

```python
flow = torch.from_numpy(flow_cv).permute(2, 0, 1).unsqueeze(0)  # (H, W, 2) to (1, 2, H, W)
flow_cv = flow[0].permute(1, 2, 0).numpy()  # (1, 2, H, W) to (H, W, 2)
```

A single-channel OpenCV image `img` of shape `(H, W)` becomes `torch.from_numpy(img).float()[None, None]`.

## Value range

Values are used as given, without normalisation. Like OpenCV, the default `eps` of
[`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] assumes intensities of roughly
0-255. `eps` regularises the 2x2 system of each pixel, whose size depends on the scale of the
values: multiplying the frames by \(s\) multiplies the Farneback determinant by \(s^4\) and the
Lucas-Kanade structure tensor by \(s^2\). With values in [0, 1], in mm/h or in dBZ, either rescale
the frames to about 0-255 or scale `eps` accordingly (by \(s^4\) for Farneback, by \(s^2\) for
Lucas-Kanade); the two give the same flow. Otherwise `eps` dominates and the flow shrinks towards
zero: a texture in [0, 1] shifted by 1.5 pixels gets a Farneback flow of about 0.0002 pixels with
the default `eps`.

```python
flow = farneback(255 * frame0, 255 * frame1)  # frames in [0, 1]
flow = farneback(frame0, frame1, eps=1e-3 / 255**4)  # the same flow
```

Missing values must be filled before estimating a flow: a NaN spreads through the filters to its
whole neighbourhood.

## Dtypes and devices

float32 and float64, on the CPU and on CUDA. The computation runs on the device of the inputs;
nothing is copied between devices. [`warp`][opticalflow_torch.warp.warp],
[`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] also accept float16 and bfloat16
inputs: they compute in float32 and return the input dtype.

The functions have no Python control flow that depends on tensor values, so they work under
`torch.compile`, from PyTorch 2.7: the compiler of earlier versions crashes on some of them or
returns wrong values without an error. A new frame size or a new value of an integer parameter
(`levels`, `iterations`, ...) can trigger a recompilation.
