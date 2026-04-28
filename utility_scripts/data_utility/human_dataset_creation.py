import argparse
import pandas as pd
import os
import shutil
import json
import numpy as np
import nibabel as nib
from PIL import Image

def argument_handler():
    parser = argparse.ArgumentParser(description="Creating blank Nifti Pipeline")
    parser.add_argument('--dicom_dataset_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/dicom_dataset.csv")
    parser.add_argument('--scan_mapping_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/scan_mapping.csv")
    parser.add_argument('--output_dir', type=str, default="/mnt/fac/CX000019_DS1/brain_vlm_human_ds")
    parser.add_argument(
        '--first_n',
        action='store_true',
        default=False,
        help=(
            'If set, skip accession-frequency ranking and simply take the first 250 '
            'rows of the dataset in their existing order. '
            'Default (flag absent) retains the original behaviour of picking the '
            'accession numbers that have the most questions first.'
        ),
    )
    return parser.parse_args()

def main(args):
    dataset = pd.read_csv(args.dicom_dataset_path)
    output_ds = pd.DataFrame(columns=dataset.columns)

    if os.path.exists(args.scan_mapping_path):
        scan_mapping = pd.read_csv(args.scan_mapping_path)
    else:
        print("Warning: scan mapping not found.")
        scan_mapping = pd.DataFrame(columns=['Deidentified_Accession_Number', 'sequence', 'image_path',"Accession_number"])
    
    # ------------------------------------------------------------------ #
    # Build the subset of rows to include in the human dataset.           #
    # Two modes are available (controlled by --first_n):                  #
    #   default  – rank accession numbers by question count (descending)  #
    #              and fill up to 250 questions from the most prolific     #
    #              sessions first.                                         #
    #   --first_n – simply take the first 250 rows of the dataset in      #
    #              their existing order, regardless of session frequency.  #
    # ------------------------------------------------------------------ #

    TARGET = 250

    if args.first_n:
        # ── Mode: first-N rows ──────────────────────────────────────────
        output_ds = dataset.head(TARGET).copy()
        unique_accessions = output_ds['Accession_number'].unique()
        print(f"--first_n mode: took first {len(output_ds)} rows covering "
              f"{len(unique_accessions)} unique accession numbers.")

        for accession_num in unique_accessions:
            accession_dir = os.path.join(args.output_dir, str(accession_num))
            os.makedirs(accession_dir, exist_ok=True)

            seq_paths = scan_mapping[scan_mapping['Accession_number'] == str(accession_num)]
            if seq_paths.empty:
                seq_paths = scan_mapping[scan_mapping['Accession_number'] == accession_num]

            for _, row_item in seq_paths.iterrows():
                image_path = row_item['image_path']
                seq_name = str(row_item['sequence']).replace('/', '_').replace('\\', '_')

                if pd.notna(image_path) and os.path.exists(str(image_path)):
                    seq_dir = os.path.join(accession_dir, seq_name)
                    os.makedirs(seq_dir, exist_ok=True)
                    dest_path = os.path.join(seq_dir, os.path.basename(str(image_path)))

                    if os.path.exists(dest_path):
                        continue

                    if os.path.isdir(str(image_path)):
                        shutil.copytree(str(image_path), dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(image_path), dest_path)
                else:
                    print(f"Warning: Image path for {accession_num} sequence {seq_name} is invalid or missing.")

    else:
        # ── Mode: most-questions-first (original behaviour) ─────────────
        freq_series = dataset['Accession_number'].value_counts(ascending=False)

        total = TARGET
        path_cnt = 0
        for accession_num, count in freq_series.items():
            if total <= 0:
                break
            path_cnt += 1

            rows = dataset[dataset['Accession_number'] == accession_num]
            if len(rows) > total:
                rows = rows.head(total)

            output_ds = pd.concat([output_ds, rows], ignore_index=True)
            total -= len(rows)

            accession_dir = os.path.join(args.output_dir, str(accession_num))
            os.makedirs(accession_dir, exist_ok=True)

            seq_paths = scan_mapping[scan_mapping['Accession_number'] == str(accession_num)]
            if seq_paths.empty:
                seq_paths = scan_mapping[scan_mapping['Accession_number'] == accession_num]

            for _, row_item in seq_paths.iterrows():
                image_path = row_item['image_path']
                # Sanitize the sequence string to be safe for filenames
                seq_name = str(row_item['sequence']).replace('/', '_').replace('\\', '_')

                if pd.notna(image_path) and os.path.exists(str(image_path)):
                    # Create a sequence subdirectory inside the accession directory
                    seq_dir = os.path.join(accession_dir, seq_name)
                    os.makedirs(seq_dir, exist_ok=True)
                    dest_path = os.path.join(seq_dir, os.path.basename(str(image_path)))

                    if os.path.exists(dest_path):
                        continue  # Skip to save time if already populated

                    # Using shutil is safer and avoids the nested folder issue when ran multiple times
                    if os.path.isdir(str(image_path)):
                        shutil.copytree(str(image_path), dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(image_path), dest_path)
                else:
                    print(f"Warning: Image path for {accession_num} sequence {seq_name} is invalid or missing.")

            if total <= 0:
                print(f"Reached target QA pairs. {path_cnt} unique accession numbers processed.")
                break

    output_ds.to_csv(os.path.join(args.output_dir, "human_dataset.csv"), index=False)

if __name__ == "__main__":
    main(argument_handler())