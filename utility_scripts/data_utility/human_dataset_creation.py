"""
human_dataset_creation.py
--------------------------
Purpose:
    Builds the image dataset used for the human QA evaluation study.

    Given a CSV of QA pairs (dicom_dataset.csv) and a scan-to-image mapping
    (scan_mapping.csv), this script:
      1. Selects a target subset of QA rows (up to 250 by default).
      2. For every unique accession number in that subset, copies the
         corresponding imaging sequences from their source locations on the
         shared filesystem into a structured output directory.
      3. Writes the filtered QA pairs to human_dataset.csv so the GUI can
         load exactly the questions that match the copied images.

    Two sampling modes are supported (controlled by --first_n):
      • first_n (default ON)  – take the first 250 rows in dataset order.
      • freq-ranked (default OFF) – greedily pick accession numbers that
          have the most questions first, filling up to 250 rows.

Output directory layout:
    <output_dir>/
        <accession_number>/
            <sequence_name>/
                <image_file_or_folder>
        human_dataset.csv
"""

import argparse
import os
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# CLI argument definition
# ---------------------------------------------------------------------------

def argument_handler():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following fields:
          - dicom_dataset_path : path to the master QA pairs CSV
          - scan_mapping_path  : path to the CSV that maps accession numbers
                                 and sequence names to image file paths
          - output_dir         : root directory where images and the output
                                 CSV will be written
          - first_n            : boolean flag selecting the sampling strategy
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the human-evaluation image dataset by selecting a subset "
            "of QA pairs and copying the associated imaging sequences."
        )
    )
    parser.add_argument(
        '--dicom_dataset_path',
        type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/dicom_dataset.csv",
        help="Path to the master QA pairs CSV (dicom_dataset.csv).",
    )
    parser.add_argument(
        '--scan_mapping_path',
        type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/scan_mapping.csv",
        help=(
            "Path to scan_mapping.csv, which maps each "
            "(Accession_number, sequence) pair to its source image path."
        ),
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/mnt/fac/CX000019_DS1/brain_vlm_human_ds",
        help="Root directory where the structured image folders will be written.",
    )
    parser.add_argument(
        '--first_n',
        action='store_true',
        default=True,
        help=(
            'If set (default), skip accession-frequency ranking and simply '
            'take the first 250 rows of the dataset in their existing order. '
            'If not set, uses the original behaviour of picking accession '
            'numbers that have the most questions first.'
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_sequence(accession_num: str, seq_name: str, image_path: str,
                   accession_dir: str) -> str:
    """
    Copy a single imaging sequence to the output directory.

    The destination is:
        <accession_dir>/<seq_name>/<basename of image_path>

    Skips the copy if the destination already exists (idempotent).
    Handles both directory trees (shutil.copytree) and plain files
    (shutil.copy2) so the same logic works for DICOM folders and NIfTI files.

    Parameters
    ----------
    accession_num : str  – used only for log messages
    seq_name      : str  – the sanitized sequence label
    image_path    : str  – source path on the shared filesystem
    accession_dir : str  – destination parent directory

    Returns
    -------
    str  – one of: 'copied', 'skipped', 'missing'
    """
    if not (pd.notna(image_path) and os.path.exists(str(image_path))):
        return 'missing'

    seq_dir   = os.path.join(accession_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    dest_path = os.path.join(seq_dir, os.path.basename(str(image_path)))

    # If the destination already exists, skip to avoid redundant I/O.
    # This makes the script safe to re-run without duplicating work.
    if os.path.exists(dest_path):
        return 'skipped'

    # Use shutil so the copy works correctly whether the source is a
    # directory tree (e.g. a DICOM series folder) or a single file (NIfTI).
    if os.path.isdir(str(image_path)):
        shutil.copytree(str(image_path), dest_path, dirs_exist_ok=True)
    else:
        shutil.copy2(str(image_path), dest_path)

    return 'copied'


def _process_accession(accession_num, scan_mapping: pd.DataFrame,
                        accession_dir: str):
    """
    Copy all imaging sequences for one accession number.

    Looks up every row in scan_mapping whose Accession_number matches
    accession_num, then calls _copy_sequence for each sequence found.

    Parameters
    ----------
    accession_num  – the accession identifier (str or int)
    scan_mapping   – the full scan_mapping DataFrame
    accession_dir  – destination directory for this accession
    """
    os.makedirs(accession_dir, exist_ok=True)

    # Filter scan_mapping to rows for this accession.
    # Try string comparison first, then fall back to native dtype in case
    # the CSV stores numbers without quotes.
    seq_paths = scan_mapping[scan_mapping['Accession_number'] == str(accession_num)]
    if seq_paths.empty:
        seq_paths = scan_mapping[scan_mapping['Accession_number'] == accession_num]

    if seq_paths.empty:
        print(f"    [WARN] No sequences found in scan_mapping for accession {accession_num}.")
        return

    copied_count  = 0
    skipped_count = 0
    missing_count = 0

    for _, row_item in seq_paths.iterrows():
        image_path = row_item['image_path']

        # Replace path separators so the sequence name is safe to use as a
        # directory name on any OS.
        seq_name = str(row_item['sequence']).replace('/', '_').replace('\\', '_')

        result = _copy_sequence(
            accession_num=str(accession_num),
            seq_name=seq_name,
            image_path=image_path,
            accession_dir=accession_dir,
        )

        if result == 'copied':
            copied_count += 1
            print(f"      [OK]      Copied  : {seq_name}")
        elif result == 'skipped':
            skipped_count += 1
            print(f"      [SKIP]    Exists  : {seq_name}")
        else:  # 'missing'
            missing_count += 1
            print(f"      [WARN]    Missing : {seq_name}  (path: {image_path})")

    print(
        f"    → {accession_num} done  "
        f"| copied={copied_count}  skipped={skipped_count}  missing={missing_count}"
    )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(args):
    """
    Entry point: load data, select rows, copy images, write output CSV.
    """
    TARGET = 250  # Maximum number of QA rows to include in the human dataset

    # ------------------------------------------------------------------
    # 1. Load the master QA pairs CSV
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"[1/4] Loading QA dataset from:\n      {args.dicom_dataset_path}")
    dataset = pd.read_csv(args.dicom_dataset_path)
    print(f"      Loaded {len(dataset):,} total QA rows.")

    # ------------------------------------------------------------------
    # 2. Load the scan-to-image mapping
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading scan mapping from:\n      {args.scan_mapping_path}")
    if os.path.exists(args.scan_mapping_path):
        scan_mapping = pd.read_csv(args.scan_mapping_path)
        print(f"      Loaded {len(scan_mapping):,} sequence entries.")
    else:
        print("      [WARN] scan_mapping.csv not found – no images will be copied.")
        scan_mapping = pd.DataFrame(
            columns=['Deidentified_Accession_Number', 'sequence', 'image_path', 'Accession_number']
        )

    # ------------------------------------------------------------------
    # 3. Select the subset of QA rows to include
    # ------------------------------------------------------------------
    print(f"\n[3/4] Selecting up to {TARGET} QA rows (mode: {'first_n' if args.first_n else 'freq-ranked'}) ...")

    if args.first_n:
        # ── Mode A: first-N rows ──────────────────────────────────────
        # Simply slice the first TARGET rows in whatever order the CSV
        # was saved. This is the default and is reproducible without
        # any sorting step.
        output_ds         = dataset.head(TARGET).copy()
        unique_accessions = output_ds['Accession_number'].unique()

        print(
            f"      Selected {len(output_ds)} rows covering "
            f"{len(unique_accessions)} unique accession number(s)."
        )

    else:
        # ── Mode B: most-questions-first (original behaviour) ─────────
        # Rank accession numbers by how many QA pairs they have
        # (descending). This ensures we fill the 250-row budget with
        # accessions that contribute the most clinical questions,
        # maximising the diversity of questions per image session seen.
        freq_series = dataset['Accession_number'].value_counts(ascending=False)

        output_ds  = pd.DataFrame(columns=dataset.columns)
        remaining  = TARGET
        path_cnt   = 0

        for accession_num, count in freq_series.items():
            if remaining <= 0:
                break

            path_cnt += 1
            rows = dataset[dataset['Accession_number'] == accession_num]

            # Cap the rows taken from this accession so we don't exceed TARGET
            if len(rows) > remaining:
                rows = rows.head(remaining)

            output_ds  = pd.concat([output_ds, rows], ignore_index=True)
            remaining -= len(rows)

        unique_accessions = output_ds['Accession_number'].unique()
        print(
            f"      Selected {len(output_ds)} rows covering "
            f"{len(unique_accessions)} unique accession number(s)."
        )

    # ------------------------------------------------------------------
    # 4. Copy imaging sequences for every selected accession number
    # ------------------------------------------------------------------
    print(f"\n[4/4] Copying images to:\n      {args.output_dir}")
    print(f"      Processing {len(unique_accessions)} accession number(s) ...\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, accession_num in enumerate(unique_accessions, start=1):
        accession_dir = os.path.join(args.output_dir, str(accession_num))
        print(
            f"  [{idx}/{len(unique_accessions)}] Accession: {accession_num}"
        )
        _process_accession(accession_num, scan_mapping, accession_dir)

    # ------------------------------------------------------------------
    # 5. Write the filtered QA pairs to disk
    # ------------------------------------------------------------------
    output_csv = os.path.join(args.output_dir, "human_dataset.csv")
    output_ds.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(f"[DONE] human_dataset.csv written ({len(output_ds)} rows) to:")
    print(f"       {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main(argument_handler())