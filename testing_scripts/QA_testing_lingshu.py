"""
QA_testing_lingshu.py — Inference for Lingshu-32B (Qwen2.5-VL based) on Brain MRI VQA.

Handles single_slice, montage_slice, and blank variants.
Uses Pydantic + JSON-constrained generation (via prompt) to ensure consistent output.
"""

import os
import pandas as pd
import torch
import argparse
import json
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
# Add project root to sys.path to import config_utils
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config

_cfg = load_config()

# --- Pydantic Schema ---
class VQAResponse(BaseModel):
    concise_reasoning: str = Field(description="Brief clinical reasoning for the choice.")
    answer: str = Field(description="The final choice selected from the options.")

FEW_SHOT_EXAMPLE = """
Example:
Question: Is there evidence of a midline shift?
(A) Yes
(B) No
(C) Indeterminate
(D) Not applicable

Response:
{
  "concise_reasoning": "The septum pellucidum is at the midline with no displacement of ventricular structures.",
  "answer": "B"
}
"""

def query_the_model(model, processor, question, patient_id, base_image_dir, image_filename="Axial.png", image_path_override=None):
    """
    Send one question + the patient's MRI slice to Lingshu-32B.
    """
    if image_path_override:
        patient_image_path = image_path_override
    else:
        patient_image_path = os.path.join(base_image_dir, str(patient_id), image_filename)
    
    if not os.path.exists(patient_image_path):
        return {
            "concise_reasoning": f"Error: Image not found at {patient_image_path}",
            "answer": "Error",
        }

    # Prepare prompt
    prompt_text = (
        "Instruction: You are a neuroradiologist. Analyze the MRI slice and provide a structured JSON response.\n"
        f"{FEW_SHOT_EXAMPLE}\n"
        "---\n"
        f"Question: {question}\n"
        "Return ONLY the JSON object."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": patient_image_path,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON from output
    try:
        # Attempt to find JSON block if model was chatty
        if "{" in output_text and "}" in output_text:
            json_str = output_text[output_text.find("{"):output_text.rfind("}")+1]
            data = json.loads(json_str)
        else:
            data = json.loads(output_text)
        
        return {
            "concise_reasoning": data.get("concise_reasoning", "No reasoning provided."),
            "answer": data.get("answer", "Unknown")
        }
    except Exception as e:
        return {
            "concise_reasoning": f"JSON Parse Error: {e}. Raw: {output_text}",
            "answer": "Error"
        }

def main(args):
    print(f"Loading model: {args.model_path}")
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading QA CSV from: {args.qa_path}")
    qa_data = pd.read_csv(args.qa_path)
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        qa_data = qa_data.head(args.limit)

    generated_answer = []
    generated_reasoning = []

    total = len(qa_data)
    for idx, row in qa_data.iterrows():
        print(f'Processing {idx+1}/{total} (ID: {row["Assigned ID"]})...')
        
        response = query_the_model(
            model, processor, 
            row["Question"], 
            row["Assigned ID"], 
            args.image_dir, 
            image_filename=args.image_filename,
            image_path_override=args.image_path
        )
        
        generated_answer.append(response['answer'])
        generated_reasoning.append(response['concise_reasoning'])
        print(f"Response: {response}\n{'-'*30}")

    qa_data["predicted_answer"] = generated_answer
    qa_data["Lingshu_Reasoning"] = generated_reasoning

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    qa_data.to_csv(args.output_path, index=False)
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu-32B VQA Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="lingshu-medical-mllm/Lingshu-32B")
    parser.add_argument('--image_dir', type=str, default=_cfg.get("slice_dir"))
    parser.add_argument('--image_path', type=str, default=None, help="Path to single image (e.g. for blank variant)")
    parser.add_argument('--image_filename', type=str, default="Axial.png")
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()
    main(args)
