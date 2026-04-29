"""
montage_slices.py  —  Multi-contrast axial montage generator

Mirrors best_slice_extraction_2d.py's data loading pattern:
  - Reads qa_path to get unique Assigned IDs (UCSF-PDGM-XXXX)
  - Constructs NIfTI directory as <nifti_root>/<pdgm_id>_nifti/
  - Loads ALL *.nii.gz files found there as contrasts
  - Picks the best axial slice from each (mask-guided or intensity CoM)
  - Assembles a labeled grid montage per patient

Usage
-----
# Uses config.yaml defaults
python data_pipeline/montage_slices.py

# Override paths
python data_pipeline/montage_slices.py \\
    --qa_path  /path/to/qa.csv \\
    --nifti_root /mnt/scratch/UCSF-PDGM-v5 \\
    --output_dir /path/to/montages \\
    --cols 4 --thumb_px 256

# Debug a single patient
python data_pipeline/montage_slices.py --pdgm_id UCSF-PDGM-0005 --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config

_cfg = load_config()

# ── Display constants ─────────────────────────────────────────────────────────
FONT_SIZE  = 18
BG_COLOR   = (10, 10, 10)
TEXT_COLOR = (220, 220, 220)
BORDER_PX  = 4


# ── Slice-selection helpers (same logic as best_slice_extraction_2d.py) ───────

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


def _extract_axial_panel(nifti_path: str,
                         mask_path: Optional[str],
                         thumb_px: int) -> Optional[Image.Image]:
    """
    Loads a NIfTI, picks the best axial slice, normalises to [0,255], and
    returns a square RGB PIL Image padded to thumb_px × thumb_px.
    """
    if not os.path.exists(nifti_path):
        return None
    try:
        img = nib.load(nifti_path).get_fdata(dtype=np.float32)
        if img.ndim == 4:
            img = img[..., 0]

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = nib.load(mask_path).get_fdata(dtype=np.float32)

        z_idx = _best_axial_index(img, mask)
        plane = img[:, :, z_idx]

        # Robust percentile normalisation (mirrors best_slice_extraction_2d.py)
        v_min, v_max = np.percentile(plane, [0.5, 99.5])
        plane = np.clip(plane, v_min, v_max)
        if v_max > v_min:
            plane = ((plane - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            plane = np.zeros_like(plane, dtype=np.uint8)

        # Rotate to standard anatomical display (same as best_slice_extraction_2d.py)
        pil_img = Image.fromarray(plane).transpose(Image.ROTATE_90).convert("RGB")
        pil_img.thumbnail((thumb_px, thumb_px), Image.LANCZOS)

        canvas = Image.new("RGB", (thumb_px, thumb_px), BG_COLOR)
        canvas.paste(pil_img, ((thumb_px - pil_img.width)  // 2,
                                (thumb_px - pil_img.height) // 2))
        return canvas

    except Exception as exc:
        print(f"  [WARN] Could not load {nifti_path}: {exc}")
        return None


# ── Label / montage helpers ───────────────────────────────────────────────────

def _get_font(size: int = FONT_SIZE):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _add_label(img: Image.Image, label: str, font) -> Image.Image:
    draw  = ImageDraw.Draw(img)
    short = textwrap.shorten(label, width=28, placeholder="…")
    y     = img.height - FONT_SIZE - 9
    draw.text((9,  y),     short, font=font, fill=(0, 0, 0))   # shadow
    draw.text((8,  y - 1), short, font=font, fill=TEXT_COLOR)
    return img


def _build_montage(panels: list[tuple[str, Image.Image]],
                   cols: int,
                   thumb_px: int,
                   title: str = "") -> Image.Image:
    font       = _get_font(FONT_SIZE)
    title_font = _get_font(FONT_SIZE + 4)
    rows       = (len(panels) + cols - 1) // cols
    title_h    = (FONT_SIZE + 16) if title else 0
    total_w    = cols * thumb_px + (cols + 1) * BORDER_PX
    total_h    = rows * thumb_px + (rows + 1) * BORDER_PX + title_h

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    if title:
        ImageDraw.Draw(canvas).text((BORDER_PX, BORDER_PX), title,
                                    font=title_font, fill=TEXT_COLOR)

    for idx, (label, panel) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x   = BORDER_PX + col * (thumb_px + BORDER_PX)
        y   = title_h + BORDER_PX + row * (thumb_px + BORDER_PX)
        canvas.paste(_add_label(panel, label, font), (x, y))

    return canvas


# ── Per-patient processing ────────────────────────────────────────────────────

# Preferred display order for well-known sequence names
_SEQUENCE_ORDER = ["T1", "T1c", "T1C", "T1CE", "T2", "FLAIR", "ADC", "DWI", "SWI"]

def _sort_key(name: str) -> tuple[int, str]:
    upper = name.upper()
    for i, s in enumerate(_SEQUENCE_ORDER):
        if s.upper() in upper:
            return (i, name)
    return (len(_SEQUENCE_ORDER), name)


def process_patient(pdgm_id: str,
                    nifti_root: str,
                    output_dir: str,
                    mask_dir: Optional[str],
                    thumb_px: int,
                    cols: int,
                    overwrite: bool) -> bool:
    """
    Generates a montage for a single patient.  Returns True if created.
    """
    out_path = Path(output_dir) / str(pdgm_id) / "montage.png"
    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {pdgm_id} — montage already exists")
        return False

    # Construct scan directory the same way as best_slice_extraction_2d.py
    scan_dir = Path(nifti_root) / f"{pdgm_id}_nifti"
    if not scan_dir.is_dir():
        print(f"  [WARN] {pdgm_id} — NIfTI directory not found: {scan_dir}")
        return False

    # Collect all NIfTI files in the scan directory
    nifti_files = sorted(
        scan_dir.glob("*.nii.gz"),
        key=lambda p: _sort_key(p.stem.replace(".nii", "").split("_")[-1])
    )
    if not nifti_files:
        print(f"  [WARN] {pdgm_id} — no *.nii.gz files found in {scan_dir}")
        return False

    # Optional tumour mask for better slice selection
    mask_path = None
    if mask_dir:
        candidate = Path(mask_dir) / f"{pdgm_id}_mask.nii.gz"
        if candidate.exists():
            mask_path = str(candidate)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Image.Image]] = []
    for nii in nifti_files:
        # Derive a human-readable label from the filename
        # e.g. UCSF-PDGM-0005_FLAIR.nii.gz → FLAIR
        stem  = nii.name.replace(".nii.gz", "").replace(".nii", "")
        label = stem.replace(f"{pdgm_id}_", "")

        panel = _extract_axial_panel(str(nii), mask_path=mask_path, thumb_px=thumb_px)
        if panel is None:
            # Placeholder for unreadable files
            panel = Image.new("RGB", (thumb_px, thumb_px), (30, 30, 30))
            ImageDraw.Draw(panel).text((8, thumb_px // 2 - 10),
                                       "MISSING", fill=(180, 60, 60))

        panels.append((label, panel))

    title   = f"Axial Montage  |  {pdgm_id}  |  {len(panels)} sequences"
    montage = _build_montage(panels, cols=cols, thumb_px=thumb_px, title=title)
    montage.save(str(out_path))
    print(f"  [OK]   {pdgm_id} → {out_path}  ({len(panels)} sequences)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Mirror best_slice_extraction_2d.py: load qa_path and get unique Assigned IDs
    qa_data = pd.read_csv(args.qa_path, on_bad_lines="skip")
    qa_data = qa_data[["Assigned ID"]].drop_duplicates()
    qa_data = qa_data.rename(columns={"Assigned ID": "Assigned_ID"})
    print(f"Loaded {len(qa_data)} unique patients from {args.qa_path}")

    # Optional: restrict to a single patient for debugging
    if args.pdgm_id:
        qa_data = qa_data[qa_data["Assigned_ID"] == args.pdgm_id]
        if qa_data.empty:
            print(f"[ERROR] '{args.pdgm_id}' not found in {args.qa_path}")
            sys.exit(1)

    created = skipped = 0
    for row in qa_data.itertuples():
        pdgm_id = row.Assigned_ID
        print(f"\nProcessing {pdgm_id} …")
        ok = process_patient(
            pdgm_id    = pdgm_id,
            nifti_root = args.nifti_root,
            output_dir = args.output_dir,
            mask_dir   = args.mask_dir,
            thumb_px   = args.thumb_px,
            cols       = args.cols,
            overwrite  = args.overwrite,
        )
        if ok:
            created += 1
        else:
            skipped += 1

    print(f"\nDone. Created {created}, skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate axial multi-contrast montages from a QA CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_path", type=str,
        default=_cfg.get("qa_path"),
        help="QA CSV with an 'Assigned ID' column (same as best_slice_extraction_2d.py).",
    )
    parser.add_argument(
        "--nifti_root", type=str,
        default=_cfg.get("nifti_root"),
        help="Root directory containing <pdgm_id>_nifti/ subdirectories.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=_cfg.get("output_base", "") + "/montages",
        help="Root directory for output montage PNGs.",
    )
    parser.add_argument(
        "--mask_dir", type=str,
        default=_cfg.get("mask_dir"),
        help="Directory with <pdgm_id>_mask.nii.gz files (from "
             "best_slice_extraction_2d.py) for mask-guided slice selection. "
             "Defaults to mask_dir in config.yaml.",
    )
    parser.add_argument(
        "--cols", type=int, default=4,
        help="Number of columns in the montage grid.",
    )
    parser.add_argument(
        "--thumb_px", type=int, default=256,
        help="Width/height of each panel thumbnail in pixels.",
    )
    parser.add_argument(
        "--pdgm_id", type=str, default=None,
        help="If set, only process this single patient (e.g. UCSF-PDGM-0005).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate montages even if they already exist.",
    )
    main(parser.parse_args())
