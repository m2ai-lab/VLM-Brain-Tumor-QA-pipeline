import numpy as np
import nibabel as nib
from PIL import Image
import argparse
import pandas as pd
import os
import torch

# MONAI Imports for Inference
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd,ConcatItemsd
)
from monai.transforms import LoadImaged

# --- CONFIGURATION ---
# Path to your downloaded weights (.pt or .pth file)
PRETRAINED_WEIGHTS_PATH = "/path/to/your/model_swinvit.pt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_model():
    """Initializes the SwinUNETR model for MONAI 1.5.2."""
    model = SwinUNETR(
        # img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    ).to(DEVICE)
    
    # Check if weights exist
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        checkpoint = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)
        # Handle different saving formats (standard MONAI vs Lightning)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Weights not found at {PRETRAINED_WEIGHTS_PATH}")
    
    model.eval()
    return model


def run_inference(model, scan_dir_path, pdgm_id, save_path):
    """
    Loads 4 separate NIfTI modalities and stacks them for SwinUNETR.
    """
    # Define the files. Adjust these filenames to match your dataset exactly.
    data = {
        "t1": os.path.join(scan_dir_path, f"{pdgm_id}_T1.nii.gz"),
        "t1c": os.path.join(scan_dir_path, f"{pdgm_id}_T1c.nii.gz"),
        "t2": os.path.join(scan_dir_path, f"{pdgm_id}_T2.nii.gz"),
        "flair": os.path.join(scan_dir_path, f"{pdgm_id}_FLAIR.nii.gz")
    }

    # Check if all files exist before processing
    for k, v in data.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"Missing modality {k} at {v}")

    transforms = Compose([
        LoadImaged(keys=["t1", "t1c", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1", "t1c", "t2", "flair"]),
        # Stacks the four 1-channel images into one 4-channel image
        ConcatItemsd(keys=["t1", "t1c", "t2", "flair"], name="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    # Execute transforms
    processed_data = transforms(data)
    inputs = processed_data["image"].unsqueeze(0).to(DEVICE) 

    # 2. Run Inference
    with torch.no_grad():
        output = sliding_window_inference(
            inputs, (128, 128, 128), 4, model, overlap=0.5
        )

    # 3. Process Output
    sigmoid_output = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = np.any(sigmoid_output > 0.5, axis=0).astype(np.uint8)

    # 4. Save using the affine from the FLAIR image to ensure alignment
    mask_nifti = nib.Nifti1Image(
        binary_mask, 
        processed_data["image"].affine.numpy()
    )
    nib.save(mask_nifti, save_path)
    return save_path

def extract_orthogonal_max_slices(image_path, mask_path):
    img_obj = nib.load(image_path)
    img = img_obj.get_fdata()
    
    mask_obj = nib.load(mask_path)
    mask = mask_obj.get_fdata()

    # Find the tumor center using the mask
    if np.sum(mask) == 0:
        print("Warning: Mask is empty. AI found no tumor.")
        # Fallback to center of the brain
        x_idx, y_idx, z_idx = np.array(img.shape) // 2
    else:
        # Get the center of mass of the tumor mask
        coords = np.argwhere(mask)
        z_idx, y_idx, x_idx = np.mean(coords, axis=0).astype(int)

    slices = {
        "axial": img[:, :, z_idx],
        "coronal": img[:, y_idx, :],
        "sagittal": img[x_idx, :, :]
    }

    processed_slices = []
    for view_name, data in slices.items():
        # --- ROBUST SCALING ---
        # 1. Ignore the extreme 1% of outliers (noise/artifacts)
        v_min, v_max = np.percentile(data, [0.5, 99.5])
        
        # 2. Clip and scale
        data = np.clip(data, v_min, v_max)
        if v_max > v_min:
            data = ((data - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
            
        # 3. Rotate to standard anatomical orientation
        img_pil = Image.fromarray(data).transpose(Image.ROTATE_90)
        processed_slices.append(img_pil)

    return processed_slices

# --- MAIN ---
def main(args):
    # 1. Initialize Model
    model = setup_model()
    
    qa_data = pd.read_csv(args.qa_path, on_bad_lines='skip')
    # Ensure columns exist
    qa_data = qa_data[['Assigned ID']].drop_duplicates()
    qa_data = qa_data.rename(columns={'Assigned ID' : 'Assigned_ID'})
    
    slice_names = ["Axial", "Coronal", "Sagittal"]
    
    # Create mask directory if not exists
    mask_dir = os.path.join(os.path.dirname(args.output_slice_path), "generated_masks")
    os.makedirs(mask_dir, exist_ok=True)

    count = 0
    for row in qa_data.itertuples():
        pdgm_id = row.Assigned_ID
        slice_dir_path = os.path.join(args.output_slice_path, str(pdgm_id))
        
        # Define where the mask SHOULD be
        mask_filename = f"{pdgm_id}_mask.nii.gz"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(slice_dir_path):
            os.makedirs(slice_dir_path, exist_ok=True)
            scan_dir_path = os.path.join(f'/mnt/fac/CX000019_DS1/UCSF-PGDM/PKG_-_UCSF-PDGM_Version_5/UCSF-PDGM-v5/{pdgm_id}_nifti')                
            # 2. GENERATE MASK IF MISSING
            if not os.path.exists(mask_path):
                print(f"Generating mask for {pdgm_id}...")
                try:
                    run_inference(model,scan_dir_path,pdgm_id, mask_path)
                except Exception as e:
                    print(f"Failed to segment {pdgm_id}: {e}")
                    continue # Skip this patient
            
            # 3. Extract Slices
            try:
                flair_path = os.path.join(scan_dir_path, f"{pdgm_id}_FLAIR.nii.gz")
                slices = extract_orthogonal_max_slices(flair_path, mask_path)
                print(f"DEBUG: Mask sum for {pdgm_id}: {np.sum(nib.load(mask_path).get_fdata())}")
                for i, s in enumerate(slices):
                    # FIX: f-string syntax and .save on the object 's'
                    s.save(os.path.join(slice_dir_path, f"{slice_names[i]}.png"))
                
                print(f"Created slices for {pdgm_id}")
                count += 1
            except Exception as e:
                print(f"Error extracting slices for {pdgm_id}: {e}")

        else:
            print(f'Slices already exist for {pdgm_id} skipping ...')

    print(f"Created {count} slice sets out of {qa_data.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best Slice Extraction")
    parser.add_argument('--qa_path', type=str, required=False,default="/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")
    parser.add_argument('--output_slice_path', type=str, required=False,default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices/")
    args = parser.parse_args()
    main(args)
