"""
extract_contrast_slices.py  —  Per-contrast, multi-axis slice extractor

For each patient in the QA CSV, loads the four target NIfTI volumes
(T1, T1c, T2, FLAIR) from the patient's <nifti_root>/<pdgm_id>_nifti/
directory, finds the best slice on each anatomical axis using mask-guided /
intensity-CoM logic, and saves 12 PNGs named:

    <output_dir>/<pdgm_id>/<axis>_<CONTRAST>.png

e.g.
    UCSF-PDGM-0005/axial_T1.png
    UCSF-PDGM-0005/axial_T1c.png
    UCSF-PDGM-0005/axial_T2.png
    UCSF-PDGM-0005/axial_FLAIR.png
    UCSF-PDGM-0005/coronal_T1.png
    ...
    UCSF-PDGM-0005/sagittal_FLAIR.png

Usage
-----
# Defaults from config.yaml
python data_pipeline/extract_contrast_slices.py

# Override paths
python data_pipeline/extract_contrast_slices.py \\
    --qa_path  /path/to/qa.csv \\
    --nifti_root /mnt/scratch/UCSF-PDGM-v5 \\
    --output_dir /path/to/2D_slices \\
    --thumb_px 256

# Debug a single patient
python data_pipeline/extract_contrast_slices.py --pdgm_id UCSF-PDGM-0005 --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config

_cfg = load_config()

# ── Target contrasts (only these four are extracted) ─────────────────────────
TARGET_CONTRASTS = ["T1", "T1c", "T2", "FLAIR"]

# Axes to extract: name → (array-dimension-index, slice-getter)
# dim 0 = X → sagittal,  dim 1 = Y → coronal,  dim 2 = Z → axial
AXES = ["axial", "coronal", "sagittal"]

# ── Display constants ─────────────────────────────────────────────────────────
BG_COLOR = (10, 10, 10)


# ── Slice-selection helpers ───────────────────────────────────────────────────

def _best_index(img: np.ndarray,
                axis: int,
                mask: Optional[np.ndarray] = None) -> int:
    """
    Return the best slice index along *axis* (0=X/sagittal, 1=Y/coronal,
    2=Z/axial).

    Priority:
      1. Centre-of-mass of a non-empty binary mask (tumour centroid).
      2. Centre-of-mass of the top-10% bright voxels.
      3. Middle slice (fallback).
    """
    size = img.shape[axis]

    if mask is not None and np.sum(mask) > 0:
        coords = np.argwhere(mask > 0.5)
        return int(np.mean(coords[:, axis]))

    nonzero = img[img > 0]
    if nonzero.size > 0:
        bright = img > np.percentile(nonzero, 90)
        if bright.sum() > 0:
            coords = np.argwhere(bright)
            return int(np.mean(coords[:, axis]))

    return size // 2


def _get_plane(img: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2-D plane from a 3-D volume at the given axis/index."""
    if axis == 0:
        return img[idx, :, :]
    elif axis == 1:
        return img[:, idx, :]
    else:
        return img[:, :, idx]


def _plane_to_png(plane: np.ndarray, thumb_px: int) -> Image.Image:
    """
    Normalise a 2-D float array to [0,255], rotate to standard anatomical
    orientation, and return a square RGB PIL Image padded to thumb_px².
    """
    v_min, v_max = np.percentile(plane, [0.5, 99.5])
    plane = np.clip(plane, v_min, v_max)
    if v_max > v_min:
        plane = ((plane - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    else:
        plane = np.zeros_like(plane, dtype=np.uint8)

    pil_img = Image.fromarray(plane).transpose(Image.ROTATE_90).convert("RGB")
    pil_img.thumbnail((thumb_px, thumb_px), Image.LANCZOS)

    canvas = Image.new("RGB", (thumb_px, thumb_px), BG_COLOR)
    canvas.paste(pil_img,
                 ((thumb_px - pil_img.width)  // 2,
                  (thumb_px - pil_img.height) // 2))
    return canvas


def _extract_slice(nifti_path: str,
                   axis: int,
                   mask: Optional[np.ndarray],
                   thumb_px: int) -> Optional[Image.Image]:
    """
    Load a NIfTI, pick the best slice on *axis*, normalise, and return
    a square RGB PIL Image.  Returns None on load failure.
    """
    if not os.path.exists(nifti_path):
        return None
    try:
        img = nib.load(nifti_path).get_fdata(dtype=np.float32)
        if img.ndim == 4:
            img = img[..., 0]

        idx = _best_index(img, axis=axis, mask=mask)
        plane = _get_plane(img, axis=axis, idx=idx)
        return _plane_to_png(plane, thumb_px)

    except Exception as exc:
        print(f"  [WARN] Could not load {nifti_path}: {exc}")
        return None


# ── Per-patient processing ────────────────────────────────────────────────────

def process_patient(pdgm_id: str,
                    nifti_root: str,
                    output_dir: str,
                    mask_dir: Optional[str],
                    thumb_px: int,
                    overwrite: bool) -> tuple[int, int]:
    """
    Extract Axial / Coronal / Sagittal PNGs for T1, T1c, T2, FLAIR.

    Returns (n_saved, n_skipped).
    """
    patient_out = Path(output_dir) / str(pdgm_id)
    scan_dir    = Path(nifti_root) / f"{pdgm_id}_nifti"

    if not scan_dir.is_dir():
        print(f"  [WARN] {pdgm_id} — NIfTI directory not found: {scan_dir}")
        return 0, 0

    # Load shared tumour mask once (used for all contrasts / axes)
    mask = None
    if mask_dir:
        candidate = Path(mask_dir) / f"{pdgm_id}_mask.nii.gz"
        if candidate.exists():
            try:
                mask = nib.load(str(candidate)).get_fdata(dtype=np.float32)
            except Exception as exc:
                print(f"  [WARN] Could not load mask {candidate}: {exc}")

    patient_out.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0

    # Axes in order: axial=2, coronal=1, sagittal=0
    axis_dims = {"axial": 2, "coronal": 1, "sagittal": 0}

    for contrast in TARGET_CONTRASTS:
        nii_path = scan_dir / f"{pdgm_id}_{contrast}.nii.gz"

        if not nii_path.exists():
            print(f"  [MISS] {contrast} — {nii_path.name} not found, skipping")
            continue

        # Pre-load volume once; extract all three axes from it
        try:
            vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]
        except Exception as exc:
            print(f"  [WARN] Could not load {nii_path.name}: {exc}")
            continue

        for axis_name, axis_dim in axis_dims.items():
            out_png = patient_out / f"{axis_name}_{contrast}.png"

            if out_png.exists() and not overwrite:
                print(f"    [SKIP] {out_png.name} already exists")
                skipped += 1
                continue

            idx   = _best_index(vol, axis=axis_dim, mask=mask)
            plane = _get_plane(vol, axis=axis_dim, idx=idx)
            img   = _plane_to_png(plane, thumb_px)

            img.save(str(out_png))
            print(f"    [OK]  {out_png.name}")
            saved += 1

    return saved, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    qa_data = pd.read_csv(args.qa_path, on_bad_lines="skip")
    qa_data = qa_data[["Assigned ID"]].drop_duplicates()
    qa_data = qa_data.rename(columns={"Assigned ID": "Assigned_ID"})
    print(f"Loaded {len(qa_data)} unique patients from {args.qa_path}")

    if args.pdgm_id:
        qa_data = qa_data[qa_data["Assigned_ID"] == args.pdgm_id]
        if qa_data.empty:
            print(f"[ERROR] '{args.pdgm_id}' not found in {args.qa_path}")
            sys.exit(1)

    print(f"Contrasts : {TARGET_CONTRASTS}")
    print(f"Axes      : {AXES}")
    print(f"Output    : {args.output_dir}\n")

    total_saved = total_skipped = 0
    for row in qa_data.itertuples():
        pdgm_id = row.Assigned_ID
        print(f"\nProcessing {pdgm_id} …")
        saved, skipped = process_patient(
            pdgm_id    = pdgm_id,
            nifti_root = args.nifti_root,
            output_dir = args.output_dir,
            mask_dir   = args.mask_dir,
            thumb_px   = args.thumb_px,
            overwrite  = args.overwrite,
        )
        total_saved   += saved
        total_skipped += skipped

    print(f"\nDone. Saved {total_saved} slices, skipped {total_skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract Axial / Coronal / Sagittal PNGs for T1, T1c, T2, FLAIR "
            "for every patient in a QA CSV. "
            "Output: <output_dir>/<pdgm_id>/<axis>_<CONTRAST>.png"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_path", type=str,
        default=_cfg.get("qa_path"),
        help="QA CSV with an 'Assigned ID' column.",
    )
    parser.add_argument(
        "--nifti_root", type=str,
        default=_cfg.get("nifti_root"),
        help="Root directory containing <pdgm_id>_nifti/ subdirectories.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=_cfg.get("slice_dir"),
        help=(
            "Root directory for per-patient PNG output. "
            "Defaults to slice_dir in config.yaml."
        ),
    )
    parser.add_argument(
        "--mask_dir", type=str,
        default=_cfg.get("mask_dir"),
        help=(
            "Directory with <pdgm_id>_mask.nii.gz files for mask-guided slice "
            "selection (from best_slice_extraction_2d.py). Optional."
        ),
    )
    parser.add_argument(
        "--thumb_px", type=int, default=256,
        help="Width/height of each output PNG in pixels.",
    )
    parser.add_argument(
        "--pdgm_id", type=str, default=None,
        help="If set, only process this single patient (e.g. UCSF-PDGM-0005).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate PNGs even if they already exist.",
    )
    main(parser.parse_args())
