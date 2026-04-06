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
    parser.add_argument('--dicom_dataset_path', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="/mnt/fac/CX000019_DS1/brain_vlm_human_ds")
    return parser.parse_args()

def main(args):

    #Load the dataset
    dataset = pd.read_csv(args.dicom_dataset_path)

    # Increment the counter for this accession number
    freq_series  = dataset['Deidentified_Accession_Number'].value_counts(ascending=False)


    print(freq_series.head(20))




        


if __name__ == "__main__":
    main(argument_handler())