import numpy as np
from PIL import Image
import os.path as path
import torch
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
from pydantic import BaseModel, Field, ValidationError
import re


slice_names = ["Axial", "Coronal", "Sagittal"]

FEW_SHOT_EXAMPLE = """
Example Request:
Question: Based on the T2/FLAIR hyperintensity, what is the most likely grade? 1) Low Grade 2) High Grade

Example Response:
{
  "reasoning": "The slices show significant mass effect and central necrosis within the T1-contrast enhancing lesion, which is highly suggestive of aggressive growth.",
  "answer": "2) High Grade"
}
"""

def clean_json_string(raw_str):
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

class MedResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step clinical observation of the MRI slices.")
    answer: str = Field(description="The final choice selected from the options.")

def process_slices(image_dir: str):
    slices = []
    for name in slice_names:
        slice_path = path.join(image_dir, f'{name}.png')
        if path.exists(slice_path):
            slices.append(Image.open(slice_path).convert("RGB"))
        else:
            print(f"Warning: {slice_path} not found.")
    return slices

def query_the_model(model, processor, question, patient_id, base_image_dir):

    patient_image_dir = path.join(base_image_dir, str(patient_id))
    if not path.exists(patient_image_dir):
        return {"reasoning": f"Directory {patient_image_dir} not found.", "answer": "Error"}

    images = process_slices(patient_image_dir)
    if len(images) == 0:
        return {"reasoning": "No slices found.", "answer": "Error"}

    prompt_text = (
        "Instruction: You are a neuroradiologist. "
        "Analyze the MRI slices and return ONLY valid JSON in the following format:\n"
        '{ "reasoning": "...", "answer": "..." }\n\n'
        f"{FEW_SHOT_EXAMPLE}\n"
        "---\n"
        f"Actual Question: {question}\n"
        "Response:"
    )

    content = [{"type": "image"}] * len(images)
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=input_text,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    new_tokens = generated[0][input_len:]
    raw_response = processor.decode(new_tokens, skip_special_tokens=True).strip()

    cleaned = clean_json_string(raw_response)

    try:
        validated = MedResponse.model_validate_json(cleaned)
        return validated.model_dump()

    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": cleaned}

    except Exception as e:
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error"}

def main(args):

    model_id = args.model_path

    print(f"Loading LLaVA-Med from {model_id}...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    qa_data = pd.read_csv(args.qa_path)

    generated_answer = []
    generated_reasoning = []

    total = len(qa_data)

    for idx, row in qa_data.iterrows():
        print(f'Processing {idx+1}/{total} (ID: {row["Assigned ID"]})...')
        response = query_the_model(
            model,
            processor,
            row["Question"],
            row["Assigned ID"],
            args.image_dir
        )

        generated_answer.append(response["answer"])
        generated_reasoning.append(response["reasoning"])
        print(f"Response: {response}\n{'-'*40}")

    qa_data["LLaVAMed_Answer"] = generated_answer
    qa_data["LLaVAMed_Reasoning"] = generated_reasoning
    qa_data.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-Med Multi-Slice Inference")

    parser.add_argument('--qa_path', type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")

    parser.add_argument('--output_path', type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVAMed/Results.csv")

    parser.add_argument('--image_dir', type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices")

    parser.add_argument('--model_path', type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/LLaVA-Med/llava-med-v1.5-mistral-7b")

    args = parser.parse_args()
    main(args)