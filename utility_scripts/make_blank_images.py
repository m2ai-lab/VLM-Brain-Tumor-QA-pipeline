import argparse
import os
import yaml
import numpy as np
import nibabel as nib
from PIL import Image
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()


def argument_handler():
    parser = argparse.ArgumentParser(description="Creating blank Nifti Pipeline")
    parser.add_argument('--blank_png_path', type=str, default=_cfg.get("blank_png"))
    parser.add_argument('--blank_nifti_path', type=str, default=_cfg.get("blank_nifti"))
    return parser.parse_args()

def main(args):


    # Create 256x256 black image
    img = Image.new('RGB', (256, 256), color='black')
    img.save(args.blank_png_path)
    print("✓ Created blank 256x256 png")

    slice_data = np.array(img)

    # Stack it 176 times to create a 3D volume (256, 256, 176)
    volume_data = np.repeat(slice_data[:, :, np.newaxis], 176, axis=2)

    # Create the NIfTI with a standard 1mm affine
    affine = np.eye(4) 
    nii_img = nib.Nifti1Image(volume_data, affine)

    # Save it
    nib.save(nii_img, args.blank_nifti_path)
    print("✓ Created blank nifti")

    #update config.yaml with the paths
    _cfg["blank_png"] = args.blank_png_path
    _cfg["blank_nifti"] = args.blank_nifti_path
    with open("config.yaml", "w") as f:
        yaml.dump(_cfg, f)
    print("✓ Updated config.yaml")

if __name__ == "__main__":
    main(argument_handler())