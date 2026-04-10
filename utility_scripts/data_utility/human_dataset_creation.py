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
    parser.add_argument('--output_dir', type=str, default="/mnt/fac/CX000019_DS1/brain_vlm_human_ds")
    return parser.parse_args()

def main(args):
    dataset = pd.read_csv(args.dicom_dataset_path)
    output_ds = pd.DataFrame(columns=dataset.columns)

    freq_series = dataset['Deidentified_Accession_Number'].value_counts(ascending=False)

    total = 250
    path_cnt = 0
    for accession_num, count in freq_series.items():
        if total <= 0:
            break
        path_cnt += 1


        rows = dataset[dataset['Deidentified_Accession_Number'] == accession_num].head(total)
        output_ds = pd.concat([output_ds, rows], ignore_index=True)
        total -= len(rows)  # use actual rows added, not `count`

        accession_dir = os.path.join(args.output_dir, str(accession_num))
        os.makedirs(accession_dir, exist_ok=True)

        # Iterate over all unique paths instead of just grabbing the 0th index
        image_paths = rows['image_path'].dropna().unique()

        for image_path in image_paths:
            if os.path.exists(image_path):
                dest_path = os.path.join(accession_dir, os.path.basename(image_path))
                # Using shutil is safer and avoids the nested folder issue when ran multiple times
                if os.path.isdir(image_path):
                    shutil.copytree(image_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(image_path, dest_path)
            else:
                print(f"Warning: Image path for {accession_num} is invalid or missing.")

        if total <= 0:
            print(f"Reached 250 entries. {path_cnt} unique accession numbers processed.")

    # Reformat how new output_ds paths are updated to accommodate multiple rows
    def construct_new_path(row):
        if pd.notna(row['image_path']):
            return os.path.join(args.output_dir, str(row['Deidentified_Accession_Number']), os.path.basename(str(row['image_path'])))
        return None

    output_ds['image_path'] = output_ds.apply(construct_new_path, axis=1)

    output_ds.to_csv(os.path.join(args.output_dir, "human_dataset.csv"), index=False)

if __name__ == "__main__":
    main(argument_handler())