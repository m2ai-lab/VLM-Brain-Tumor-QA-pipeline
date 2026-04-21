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
    parser.add_argument('--blank_image_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/blank/BlackedOut.png")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/blank/BlackedOut.nii")
    return parser.parse_args()

def main(args):


    # Load your 256x256 blank PNG
    img_png = Image.open(args.blank_image_path).convert('L')
    slice_data = np.array(img_png)

    # Stack it 176 times to create a 3D volume (256, 256, 176)
    volume_data = np.repeat(slice_data[:, :, np.newaxis], 176, axis=2)

    # Create the NIfTI with a standard 1mm affine
    affine = np.eye(4) 
    nii_img = nib.Nifti1Image(volume_data, affine)

    # Save it
    nib.save(nii_img, args.output_path + ".gz")


if __name__ == "__main__":
    main(argument_handler())