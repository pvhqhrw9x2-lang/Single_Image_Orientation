# Single Image Orientation

This project implements single-image orientation techniques for estimating camera position and orientation from 2D–3D correspondences.  
It provides Python implementations of three classical photogrammetric approaches:
- **DLT (Direct Linear Transformation)**
- **PnPf (Perspective-n-Point with unknown focal length)**
- **Spatial Resection**

The workflow supports camera calibration experiments, orientation recovery, and validation through reprojection error analysis.

---

## Project Structure

| File | Description |
|------|--------------|
| `main.py` | Entry point that runs the full orientation pipeline using user-defined input and method selection. |
| `spatial_resection.py` | Implements the non-linear spatial resection algorithm to estimate exterior orientation parameters (EOPs). |
| `DLT_USV.py` | Contains the Direct Linear Transformation solution with normalization and least-squares refinement. |
| `estimate_focal.py` | Provides PnPf implementation to estimate camera pose with unknown focal length. |
| `R_decom.py` | Handles rotation matrix decomposition (e.g., RQ or Euler angles) for extracting ω–φ–κ or yaw–pitch–roll parameters. |
| `image.py` | Handles image coordinate operations and conversion between pixel and metric units. |
| `reproj_utils.py` | Computes reprojection errors and visualization utilities. |

---


