import numpy as np
import os.path as path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id
_cfg = load_config()

def query_the_model(model, tokenizer, question, image_path):
    full_image_path = image_path
    
    if not path.exists(full_image_path):
        return f"Error: Path {full_image_path} not found."

    device = torch.device("cuda")
    dtype = torch.bfloat16 

    # 1. LOAD RAW NIFTI
    image_np = sitk.GetArrayFromImage(sitk.ReadImage(full_image_path))
    
    # 2. CONVERT TO FLOAT32 TENSOR FOR INTERPOLATION
    # Shape from SimpleITK is usually (D, H, W). 
    # We need (Batch=1, Channel=1, D, H, W)
    image_pt = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)

    # 3. RESIZE TO THE MODEL'S NATIVE RESOLUTION
    # target_size is (Depth, Height, Width)
    target_size = (128, 256, 256)
    image_pt = F.interpolate(image_pt, size=target_size, mode='trilinear', align_corners=False)
    
    # 4. PREPARE FOR MODEL
    image_pt = image_pt.to(dtype=dtype, device=device)

    # 5. SETUP TEXT INPUTS
    proj_out_num = getattr(model.get_model().config, "proj_out_num", 256)
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

    generation = model.generate(
        images=image_pt,
        inputs=input_id,
        max_new_tokens=512,
        do_sample=False,
        temperature=0,
    )

    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
    return generated_texts[0]

def main(args):

    device = torch.device("cuda")  
    dtype = torch.bfloat16 

    model_id = args.model_path
    
    print(f"Loading Med3DVLM from {model_id}...")
    # Using bfloat16 is highly recommended for MedGemma if your GPU supports it
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    qa_data = pd.read_csv(args.qa_path)
    total = qa_data.shape[0]

    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    print(f"Running Med3DVLM blank inference (batch_size=1) on {total} rows.")

    for idx, row in qa_data.iterrows():
        patient_id = str(row["Assigned ID"])
        if get_row_id(patient_id, row["Question"]) in completed_ids:
            continue

        print(f'Processing {idx+1}/{total} (ID: {patient_id})...')
        response = query_the_model(model, tokenizer, row["Question"], args.image_path)
        print(f"Response: {response}\n{'-'*30}")

        save_checkpoint(
            args.output_path,
            qa_data.iloc[[idx]],
            {"predicted_answer": [response]},
        )
        completed_ids.add(get_row_id(patient_id, row["Question"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Med3DVLM Blank-Control Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, default=_cfg.get("output_base", "") + "/Med3DVLM/blank_results.csv")
    parser.add_argument('--image_path', type=str, default=_cfg.get("blank_nifti"))
    parser.add_argument('--model_path', type=str, default=_cfg.get("med3dvlm_model_path"))
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Fixed at 1: each volumetric NIfTI uses ~1 GB GPU RAM.")

    args = parser.parse_args()
    main(args)