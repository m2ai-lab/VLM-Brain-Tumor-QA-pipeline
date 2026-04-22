import numpy as np
import nibabel as nib
from PIL import Image
import os.path as path
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText 
import argparse
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import re
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

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
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

class MedResponse(BaseModel):
    # Literal ensures the answer MUST be one of these specific strings
    reasoning: str = Field(description="Step-by-step clinical observation of the MRI slices.")
    answer: str = Field(description="The final choice selected from the options.")


def query_the_model(model, processor, question, patient_id, base_image_dir):
    # Assume the PNGs are stored in a folder named after the patient ID
    patient_image_path = path.join(base_image_dir, str(patient_id),"Axial.png")
    
    if not path.exists(patient_image_path):
        return {"reasoning": f"Error: Directory {patient_image_path} not found.", "answer": "Error"}

    # 1. Get processed PIL images
    image = Image.open(patient_image_path).convert("RGB")
    
    # 1. More aggressive prompt with a clear JSON schema
    prompt_text = (
        "Instruction: You are a neuroradiologist. Analyze the MRI slice and provide a structured JSON response.\n"
        f"{FEW_SHOT_EXAMPLE}"
        "---\n"
        f"Actual Question: {question}\n"
        "Response:"
    )

    # 2. Build messages
    content = [{"type": "image"}] 
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    # 3. Process inputs
    # Use add_generation_prompt=False because we are manually adding the "{"
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Check if the template ends with <|im_start|>assistant or similar; 
    # we need to ensure our '{' comes AFTER the assistant tag.
    inputs = processor(
        text=input_text,
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # 4. Generate
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_sequence = model.generate(
            **inputs, 
            do_sample=False, 
            max_new_tokens=512,
            stop_strings=["}"], # Stop as soon as the JSON closes
            tokenizer=processor.tokenizer
        )

    # 5. Extract and RE-ADD the opening brace
    new_tokens = generated_sequence[0][input_len:]
    raw_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Since we pre-filled '{', the model output likely starts with "reasoning": ...
    full_json_str = "{" + raw_response 
    if not full_json_str.endswith("}"):
        full_json_str += "}"
    
    cleaned_response = clean_json_string(raw_response) 
    
    try:
        # Extract JSON string using regex
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model output")
        
        # Validate against the Pydantic schema
        validated_data = MedResponse.model_validate_json(json_match.group(0))
        
        return validated_data.model_dump()
        
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": cleaned_response}
    except Exception as e:
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error"}

def main(args):
    model_id = args.model_path
    
    print(f"Loading MedGemma from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Using bfloat16 is highly recommended for MedGemma if your GPU supports it
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )

    qa_data = pd.read_csv(args.qa_path)
    generated_answer = []
    generated_reasoning = []
    total = qa_data.shape[0]

    for idx, row in qa_data.iterrows():
        print(f'Processing {idx+1}/{total} (ID: {row["Assigned ID"]})...')
        response = query_the_model(model, processor, row["Question"], row["Assigned ID"], args.image_dir)
        generated_answer.append(response['answer'])
        generated_reasoning.append(response['reasoning'])
        print(f"Response: {response}\n{'-'*30}")
        
    # Save results
    qa_data["predicted_answer"] = generated_answer
    qa_data["MedGemma_Reasoning"] = generated_reasoning
    qa_data.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma NIfTI Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, default=_cfg.get("output_base", "") + "/MedGemma1.5/single_slice_results.csv")
    parser.add_argument('--image_dir', type=str, default=_cfg.get("slice_dir"))
    parser.add_argument('--model_path', type=str, default=_cfg.get("medgemma_model_path"))

    args = parser.parse_args()
    main(args)