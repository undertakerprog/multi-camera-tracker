# Jetson Camera Inventory

Inventory collected over SSH on 2026-10-27. Device names and network addresses
describe this connection session and can change after reconnecting hardware.

## Host

- SSH alias: `jetson`.
- Project: `/home/jetson/Documents/multi-camera-tracker`.
- Board: NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super.
- OS kernel: `5.15.148-tegra`, aarch64.
- Python: 3.10.12; OpenCV: 4.5.4 with GTK3.
- Basler pylon and Python `pypylon` are installed.

## Visible-Light Cameras

| Model | Serial | USB ID | Connection |
| --- | --- | --- | --- |
| Basler daA1600-60um | 24802255 | 2676:ba03 | USB 3, 5000 Mbit/s |
| Basler daA1600-60um | 24802269 | 2676:ba03 | USB 3, 5000 Mbit/s |

Both cameras share a USB 3 hub connected at 10000 Mbit/s. Neither camera
appears as `/dev/video*`; the application accesses them through pypylon.

Each camera was tested separately with the application's capture-only benchmark:

- Mono8, 1600 x 1200, exposure 3000 us, gain 0, automatic exposure/gain off.
- 183 frames in approximately 3 seconds, approximately 60.9 FPS, zero dropped
  frames during each short run.
- Both streams have been verified separately; simultaneous capture has not
  been tested.

The previous camera serial `24802261` is absent. A configured serial must be
updated, or set to `null` to open the first enumerated Basler camera.

## Other USB Devices / Thermal Camera Candidates

Two USB devices identify themselves as `0525:a4a2`, RNDIS/Ethernet Gadget,
running Linux `4.11.12-bone-rt-r3`. These are USB network devices rather than
UVC video devices. USB descriptors alone do not identify their camera models
or prove that they are the thermal imagers.

| USB path | Network interface | Observed state |
| --- | --- | --- |
| 1-2.3 | enx32700518ff78 | UP, 192.168.7.20/24 |
| 1-2.4 | enx32700518ff76 | DOWN |

On the active link, `192.168.7.2` answers ping and accepts SSH on TCP 22.
TCP ports 80, 443, 554 and 8080 refused connections. No remote login to that
device was attempted. Access to the other device has not been configured.

To implement thermal capture, identify the device models and obtain their
stream protocol or SDK. Network addressing for both devices must also be
checked before enabling simultaneous access.

## Application Status

The current application opens one Basler stream. It does not yet capture or
display four simultaneous streams and has no source for these USB network
devices. Supplying four connected devices does not itself enable multi-camera
operation.

A short GUI run on the Jetson desktop succeeded with `--serial 24802255`.
The first frame was very dark (min 0, max 5, mean 0.7 at exposure 3000 us);
check the lens cover, scene lighting and exposure before interpreting a dark
preview as a capture failure. The test was stopped intentionally.

## Brightness and Orientation Check

A subsequent check with blinds open confirmed that raw Mono8 data itself was
dark. Both internal `GreyDiagonalSawtooth8` test patterns spanned 0..255,
indicating that the transport preserved the full 8-bit intensity range.

For serial 24802255, mean brightness increased from approximately 6 at 10000 us /
0 dB to 89 at 30000 us / 12 dB. For 24802269, the same settings increased mean
brightness from approximately 2 to 36. Measurements depend on scene lighting.
Captured images also showed substantial optical defocus; exposure changes do
not resolve that blur.

One-shot exposure and gain adjustment was tested with an exposure ceiling of
30000 us. This ceiling limits exposure-related FPS loss; a camera using the full
ceiling runs at approximately 33 FPS. Software rotation by 180 degrees is now
configurable for the reported reversed horizontal and vertical image directions.
