# Rolling-Shutter-Aware Observation Modeling for 3D Gaussian Human Avatars

This repository contains the implementation and data-processing pipeline for **rolling-shutter-aware observation modeling** in 3D Gaussian human avatar reconstruction.

The project is built on top of Gaussian-avatar-based dynamic human reconstruction and extends the observation model from a **global-shutter (GS)** assumption to a **rolling-shutter (RS)** setting. The key idea is simple: keep the original avatar representation, motion prior, and optimization structure, and modify only the image formation process so that training observations better match the row-dependent timing of RS cameras.

> **Current project scope**
>
> This repository currently supports:
> - rolling-shutter data synthesis from sharp ZJU-MoCap sequences,
> - RS-aware training integration for Gaussian human avatars,
> - benchmark construction and dataset-level quantitative analysis.
>
> At this stage, the released quantitative tables primarily compare `blur11` and `rs11` against the sharp center reference. A complete end-to-end rendered-avatar-versus-ground-truth reconstruction benchmark is left for future work.

---

## Overview

Dynamic human avatar reconstruction methods often assume either:
1. sharp observations, or
2. blur that can still be explained under a GS image formation model.

This assumption breaks under rolling shutter. In RS cameras, different image rows are captured at different times. Under fast articulated human motion, this causes not only motion blur, but also **row-dependent geometric distortion**.

This repository studies that mismatch directly by replacing the original frame-wise GS observation with a **row-conditioned RS observation model** while preserving:
- Gaussian avatar representation,
- SMPL-based motion prior,
- spline-based motion modeling,
- original optimization losses.

---

## What is included

### 1. RS-aware observation formulation
The framework keeps the original dynamic avatar representation and modifies only the observation model:
- GS: frame-wise temporal averaging
- RS: row-conditioned composition according to row-dependent sampling times

### 2. Rolling-shutter synthesis pipeline
A controlled RS benchmark is constructed from **ZJU-MoCap** sharp sequences:
- sharp RGB sequences are used as source data,
- human masks are read together with RGB frames,
- different rows are assigned different sampling times within a readout window,
- optional row exposure integration is supported,
- center-time SMPL parameters and vertices are preserved for compatibility with the original pipeline.

### 3. Training integration
The synthesized RS dataset is aligned with the original blur-format directory structure so that downstream training code requires only minimal modification.

### 4. Dataset-level quantitative analysis
Current results show that rolling shutter changes the **spatial error profile** rather than acting as a uniform degradation:
- compared with `blur11`, `rs11` yields lower mean PSNR,
- but also lower foreground MAE,
- and the effect varies across scenes.

---
The experimental data is available at this link：https://drive.google.com/file/d/1eHgdeu3gg-mtiW-zUjzM_EFNtIQ6lQhO/view?usp=drive_link
## Repository structure

```text
.
├── assets/                      # SMPL files
├── data/
│   ├── BlurZJU/
│   │   ├── sharp/               # rearranged sharp sequences
│   │   ├── blur11/              # original blur benchmark
│   │   └── rs11/                # synthesized rolling-shutter benchmark
│   └── ...
├── synthesize_rolling_shutter.py
├── train.py
├── train_BlurZJU.sh
├── train_RSZJU.sh
├── train_BSHuman.sh
└── README.md


