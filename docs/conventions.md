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

[`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] (and their modules) take the span of
the frame values as `data_range`, 255 by default:

```python
flow = farneback(frame0, frame1)  # frames in 0-255
flow = farneback(frame0, frame1, data_range=1.0)  # frames in [0, 1]
flow = lucas_kanade(frame0, frame1, data_range=60.0)  # reflectivity in 0-60 dBZ
```

`eps` regularises the 2x2 system of each pixel, whose size depends on the scale of the values:
multiplying the frames by \(s\) multiplies the Farneback determinant by \(s^4\) and the
Lucas-Kanade structure tensor by \(s^2\). So `eps` is given for frames spanning 0-255, where
OpenCV's fixed value applies, and on frames spanning `data_range` it acts as
`eps * (data_range / 255) ** 4` (Farneback) or `eps * (data_range / 255) ** 2` (Lucas-Kanade).
Frames scaled together with `data_range` give the same flow, and gradients scaled inversely: the
frames are brought to 0-255 internally, so nothing but rounding depends on the scale. A learned
`eps` keeps its meaning for any `data_range`. The confidence output is that of the frames as given:
it scales with the square of the frame values, not with `data_range`.

`data_range` sets how much contrast counts as texture, so give the span of the values: too large a
value shortens the flows. Farneback, whose `eps` goes with the fourth power, is the most sensitive:
on radar frames, its flows were 6-7% too short with `data_range` twice too large, and a third too
short for frames in 0-60 dBZ left at the default. Frames in [0, 1] left at the default get a
Farneback flow of about zero (0.0002 pixels for a texture shifted by 1.5 pixels, as with OpenCV),
but a Lucas-Kanade flow held back only where the texture is weak, which is easy to miss: on the
same radar frames, 9% too short on average and 1.1 pixels off.

The default is 255 rather than 1 because the opposite mistake is worse: 0-255 frames with
`data_range=1.0` have no regularisation where they are flat. On radar frames, Farneback then gave
flows of up to 300 pixels in dry areas and gradients with respect to the frames of up to 3e8,
against about 25 pixels and 1e-3 with the default.

Missing values must be filled before estimating a flow: a NaN spreads through the filters to its
whole neighbourhood.

## Dtypes and devices

float32 and float64, on the CPU and on CUDA. The computation runs on the device of the inputs;
nothing is copied between devices. [`warp`][opticalflow_torch.warp.warp],
[`farneback`][opticalflow_torch.farneback.farneback] and
[`lucas_kanade`][opticalflow_torch.lucas_kanade.lucas_kanade] also accept float16 and bfloat16
inputs: they compute in float32 and return the input dtype.

Inside a `torch.autocast` region every function and module gives the same result as outside it:
autocast is turned off for the device of the inputs while it runs. Otherwise float16 autocast would
make the convolutions of 0-255 frames overflow, and Lucas-Kanade flows NaN. Nor do they promote
mixed dtypes as autocast does: the frames and flows of a call must share one dtype, so cast a
model's float16 output with `.float()` before passing it with float32 frames or flows.

The functions have no Python control flow that depends on tensor values, so they work under
`torch.compile`, from PyTorch 2.7: the compiler of earlier versions crashes on some of them or
returns wrong values without an error. A new frame size or a new value of an integer parameter
(`levels`, `iterations`, ...) can trigger a recompilation.
