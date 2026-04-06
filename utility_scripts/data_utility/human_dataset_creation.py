import argparse
import pandas as pd
import os
from collections import defaultdict
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

    output_ds = pd.DataFrame()
    #Load the dataset
    dataset = pd.read_csv(args.dicom_dataset_path)

    output_ds = pd.DataFrame(columns=dataset.columns)

    # Increment the counter for this accession number
    freq_series  = dataset['Deidentified_Accession_Number'].value_counts(ascending=False)

    total = 250
    path_cnt = 0
    for accession_num, count in freq_series.items():
        path_cnt += 1
        output_ds = pd.concat([output_ds, dataset[dataset['Deidentified_Accession_Number'] == accession_num]].head(total - count), ignore_index=True)
        total -= count
        # Create a directory for this accession number
        accession_dir = os.path.join(args.output_dir, str(accession_num))
        os.makedirs(accession_dir, exist_ok=True)

        # Get the image path for this accession number
        image_path = dataset.loc[dataset['Deidentified_Accession_Number'] == accession_num, 'image_path'].values[0]

        # Copy the DICOM file to the new directory
        if pd.notna(image_path) and os.path.exists(image_path):
            dest_path = os.path.join(accession_dir, os.path.basename(image_path))
            os.system(f"cp {image_path} {dest_path}")
        else:
            print(f"Warning: Image path for accession number {accession_num} is invalid or does not exist.")
        if total <= 0:
            print(f"Reached 250 entries, stopping the copying process.{path_cnt} unique accession numbers processed.")
            break

    output_ds['image_path'] = output_ds['Deidentified_Accession_Number'].apply(lambda x: os.path.join(args.output_dir, str(x)))

    output_ds.to_csv(os.path.join(args.output_dir, "human_dataset.csv"), index=False)
    





        


if __name__ == "__main__":
    main(argument_handler())