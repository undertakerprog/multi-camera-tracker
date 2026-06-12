# GPU / CUDA / TensorRT on Jetson

This document explains how the project relates to GPU acceleration on the
NVIDIA Jetson and how to gather real numbers. **No GPU acceleration is enabled
by default**, because nothing here has a measured win yet and the trackers in
use are CPU-only.

## How to diagnose the board

Run on the Jetson:

```bash
python3 tools/jetson_diagnostics.py
```

This probes and records:

- the exact board (`/proc/device-tree/model`);
- the OpenCV version, whether it was **built with CUDA**, and the CUDA device
  count (`cv2.cuda.getCudaEnabledDeviceCount()`);
- whether the contrib tracking module (CSRT/KCF) is present;
- whether **TensorRT** and **pyCUDA** import.

Results are written to `docs/JETSON_DIAGNOSTICS.md` (generated, board-specific).

## What does NOT run on the GPU

- **CSRT and KCF** (OpenCV trackers) are **CPU-only**. There is no CUDA
  implementation of these trackers, so the presence of a CUDA device does not
  make them faster. The big tracking speed-up in this project comes from running
  the tracker inside a small dynamic ROI (`DynamicROITracker`), not from CUDA.
- The Kalman filter, template matching and ORB used here run on the CPU.

## What could justifiably move to the GPU

Only after the benchmark shows a real bottleneck and only if OpenCV was built
with CUDA (`Built with CUDA: yes`, devices > 0):

- `resize` / preprocessing of the full frame via `cv2.cuda.resize` /
  `cv2.cuda_GpuMat`;
- global ORB / template search over the full frame (the expensive reacquire
  path), if it becomes a hotspot.

Measure before/after with:

```bash
python3 main.py --benchmark --benchmark-roi 600,450,80,80
```

## TensorRT detector

TensorRT is the intended runtime for the optional detector (project Stage 4).
The integration point is `src/detection/tensorrt_detector.py` — a disabled stub
today. Do **not** load an arbitrary model before the target object classes are
known. The pipeline runs detection at most once every *N* frames and relies on
the tracker/Kalman in between, so adding a detector does not change the tracking
loop.

## Rule of thumb

Do not add a heavy GPU dependency without a measured improvement from the
benchmark. Correctness, reliability and low latency come first.
