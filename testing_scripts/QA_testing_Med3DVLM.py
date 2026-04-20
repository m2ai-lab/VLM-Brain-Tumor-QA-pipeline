import numpy as np
import os.path as path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import SimpleITK as sitk
import torch.nn.functional as F

def query_the_model(model, tokenizer, question, patient_id, image_dir):
    full_image_path = path.join(image_dir, f'{patient_id}_nifti', f'{patient_id}_FLAIR.nii.gz')
    
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
    responses = []
    total = qa_data.shape[0]

    for idx, row in qa_data.iterrows():
        print(f'Processing {idx+1}/{total} (ID: {row["Assigned ID"]})...')
        response = query_the_model(model,tokenizer, row["Question"], row["Assigned ID"], args.image_dir)
        responses.append(response)
        print(f"Response: {response}\n{'-'*30}")
        

    # Save results
    qa_data["predicted_answer"] = responses
    qa_data.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medgemma QA testing")
    parser.add_argument('--qa_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/Med3DVLM/Full_nifti_results.csv")
    parser.add_argument('--image_dir', type=str, default="/scratch/user/shghosh/UCSF-PDGM-v5/")
    parser.add_argument('--model_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B")
    
    args = parser.parse_args()
    main(args)