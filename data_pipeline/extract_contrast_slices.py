"""
extract_contrast_slices.py  —  Per-contrast axial slice extractor

For each patient in the QA CSV, loads every *.nii.gz file in the patient's
<nifti_root>/<pdgm_id>_nifti/ directory, extracts the best axial slice for
each contrast (using the same mask-guided / intensity-CoM logic as
montage_slices.py), and saves individual PNGs named:

    <output_dir>/<pdgm_id>/axial_<CONTRAST>.png

e.g.
    UCSF-PDGM-0005/axial_FLAIR.png
    UCSF-PDGM-0005/axial_T1.png
    UCSF-PDGM-0005/axial_T1c.png
    UCSF-PDGM-0005/axial_T2.png
    ...

The output directory and per-patient folder layout mirror the existing
slice_dir / 2D_slices structure so the files can be used directly by the
QA testing scripts that already expect per-patient image directories.

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

# ── Display constants (match montage_slices.py) ───────────────────────────────
BG_COLOR = (10, 10, 10)


# ── Slice-selection helpers (identical to montage_slices.py) ──────────────────

def _best_axial_index(img: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    """
    Returns the axial (Z-axis) slice index that best shows the lesion.

    Priority:
      1. Centre-of-mass of a non-empty binary mask (tumour centroid).
      2. Centre-of-mass of the top-10% bright voxels.
      3. Middle slice (fallback).
    """
    z_size = img.shape[2]

    if mask is not None and np.sum(mask) > 0:
        coords = np.argwhere(mask > 0.5)
        return int(np.mean(coords[:, 2]))

    nonzero = img[img > 0]
    if nonzero.size > 0:
        bright = img > np.percentile(nonzero, 90)
        if bright.sum() > 0:
            coords = np.argwhere(bright)
            return int(np.mean(coords[:, 2]))

    return z_size // 2


def _extract_axial_slice(nifti_path: str,
                         mask_path: Optional[str],
                         thumb_px: int) -> Optional[Image.Image]:
    """
    Load a NIfTI, pick the best axial slice, normalise to [0,255], and
    return a square RGB PIL Image padded to thumb_px × thumb_px.

    Returns None if the file cannot be loaded.
    """
    if not os.path.exists(nifti_path):
        return None
    try:
        img_obj = nib.load(nifti_path)
        img_obj = nib.as_closest_canonical(img_obj)
        img = img_obj.get_fdata(dtype=np.float32)
        if img.ndim == 4:
            img = img[..., 0]

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_obj = nib.load(mask_path)
            mask_obj = nib.as_closest_canonical(mask_obj)
            mask = mask_obj.get_fdata(dtype=np.float32)

        z_idx = _best_axial_index(img, mask)
        
        # In canonical RAS, axis 2 is Axial (I-S). 
        # Plane [:, :] is X-Y. Rot90 puts Anterior (Y) at top.
        plane = img[:, :, z_idx]
        plane = np.rot90(plane)

        # Robust percentile normalisation (mirrors montage_slices.py)
        v_min, v_max = np.percentile(plane, [0.5, 99.5])
        plane = np.clip(plane, v_min, v_max)
        if v_max > v_min:
            plane = ((plane - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            plane = np.zeros_like(plane, dtype=np.uint8)

        # Standard anatomical display
        pil_img = Image.fromarray(plane).convert("RGB")
        pil_img.thumbnail((thumb_px, thumb_px), Image.LANCZOS)

        # Centre on a square canvas
        canvas = Image.new("RGB", (thumb_px, thumb_px), BG_COLOR)
        canvas.paste(pil_img, ((thumb_px - pil_img.width)  // 2,
                                (thumb_px - pil_img.height) // 2))
        return canvas

    except Exception as exc:
        print(f"  [WARN] Could not load {nifti_path}: {exc}")
        return None


# ── Contrast label extraction ─────────────────────────────────────────────────

def _contrast_label(nii_path: Path, pdgm_id: str) -> str:
    """
    Derive a clean contrast label from a NIfTI filename.

    Examples:
      UCSF-PDGM-0005_FLAIR.nii.gz  →  FLAIR
      UCSF-PDGM-0005_T1c.nii.gz    →  T1c
      UCSF-PDGM-0005_seg.nii.gz    →  seg

    Falls back to the full stem if the patient-ID prefix isn't found.
    """
    stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
    # Strip the patient-ID prefix (e.g. "UCSF-PDGM-0005_")
    prefix = f"{pdgm_id}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


# ── Per-patient processing ────────────────────────────────────────────────────

def process_patient(pdgm_id: str,
                    nifti_root: str,
                    output_dir: str,
                    mask_dir: Optional[str],
                    thumb_px: int,
                    overwrite: bool,
                    skip_seg: bool) -> tuple[int, int]:
    """
    Extract one axial PNG per contrast for a single patient.

    Returns (n_saved, n_skipped).
    """
    patient_out = Path(output_dir) / str(pdgm_id)

    # Locate the NIfTI directory
    scan_dir = Path(nifti_root) / f"{pdgm_id}_nifti"
    if not scan_dir.is_dir():
        print(f"  [WARN] {pdgm_id} — NIfTI directory not found: {scan_dir}")
        return 0, 0

    # Collect all NIfTI volumes
    nifti_files = sorted(scan_dir.glob("*.nii.gz"))
    if not nifti_files:
        print(f"  [WARN] {pdgm_id} — no *.nii.gz files found in {scan_dir}")
        return 0, 0

    # Optional tumour mask for better slice selection
    mask_path = None
    if mask_dir:
        candidate = Path(mask_dir) / f"{pdgm_id}_mask.nii.gz"
        if candidate.exists():
            mask_path = str(candidate)

    patient_out.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0
    for nii in nifti_files:
        label = _contrast_label(nii, pdgm_id)

        # Optionally skip segmentation masks (they look blank / wrong when normalised)
        if skip_seg and label.lower() in {"seg", "segmentation", "mask", "label"}:
            print(f"    [SKIP] {label} — segmentation file skipped (--skip_seg)")
            skipped += 1
            continue

        out_png = patient_out / f"axial_{label}.png"

        if out_png.exists() and not overwrite:
            print(f"    [SKIP] axial_{label}.png already exists")
            skipped += 1
            continue

        slice_img = _extract_axial_slice(str(nii), mask_path=mask_path, thumb_px=thumb_px)

        if slice_img is None:
            # Write a placeholder so the slot is visible in results
            placeholder = Image.new("RGB", (thumb_px, thumb_px), (30, 30, 30))
            placeholder.save(str(out_png))
            print(f"    [WARN] axial_{label}.png — could not load NIfTI; placeholder saved")
        else:
            slice_img.save(str(out_png))
            print(f"    [OK]  axial_{label}.png")

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
            skip_seg   = args.skip_seg,
        )
        total_saved   += saved
        total_skipped += skipped

    print(f"\nDone. Saved {total_saved} slices, skipped {total_skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract one axial PNG per contrast type for every patient in a QA CSV. "
            "Output: <output_dir>/<pdgm_id>/axial_<CONTRAST>.png"
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
            "Defaults to slice_dir in config.yaml so files land alongside "
            "existing Axial.png slices."
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
    parser.add_argument(
        "--skip_seg", action="store_true", default=True,
        help=(
            "Skip segmentation/mask volumes (files labelled seg, segmentation, "
            "mask, or label) — they render incorrectly when intensity-normalised."
        ),
    )
    main(parser.parse_args())
