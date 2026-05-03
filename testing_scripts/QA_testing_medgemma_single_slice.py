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
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id
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


def _build_prompt(question: str) -> str:
    return (
        "Instruction: You are a neuroradiologist. Analyze the MRI slice and provide a structured JSON response.\n"
        f"{FEW_SHOT_EXAMPLE}"
        "---\n"
        f"Actual Question: {question}\n"
        "Response:"
    )


def _parse_response(raw_response: str) -> dict:
    """Decode one raw generated string into a structured dict."""
    cleaned_response = clean_json_string(raw_response)
    try:
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model output")
        validated_data = MedResponse.model_validate_json(json_match.group(0))
        return validated_data.model_dump()
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": cleaned_response}
    except Exception as e:
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error"}


def run_batch(model, processor, batch_rows: list[dict], base_image_dir: str, image_filename: str) -> list[dict]:
    """
    Run inference on a batch of QA rows.

    Each element of batch_rows is a dict with at least 'Question' and 'Assigned ID'.
    Returns a list of response dicts in the same order.  Rows whose image is
    missing are returned immediately with an Error response without blocking the
    rest of the batch.
    """
    # Split into valid and missing-image rows so we don't drop the batch on one
    # bad patient directory.
    valid_indices: list[int] = []
    results: list[dict] = [{"reasoning": "Not processed", "answer": "Error"}] * len(batch_rows)

    texts: list[str] = []
    images_batch: list[Image.Image] = []

    for i, row in enumerate(batch_rows):
        patient_image_path = path.join(base_image_dir, str(row["Assigned ID"]), image_filename)
        if not path.exists(patient_image_path):
            results[i] = {"reasoning": f"Error: Image {patient_image_path} not found.", "answer": "Error"}
            continue

        img = Image.open(patient_image_path).convert("RGB")
        prompt_text = _build_prompt(row["Question"])

        content = [{"type": "image"}, {"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": content}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        valid_indices.append(i)
        texts.append(input_text)
        images_batch.append(img)

    if not valid_indices:
        return results

    # Left-pad so generation doesn't get confused by padding on the right
    processor.tokenizer.padding_side = "left"

    # Gemma3 processor requires images as a nested list:
    # [[img_for_text_0], [img_for_text_1], ...]
    # A flat list is treated as a single-image batch and causes a size mismatch.
    inputs = processor(
        text=texts,
        images=[[img] for img in images_batch],
        padding=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_sequences = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            stop_strings=["}"],
            tokenizer=processor.tokenizer,
        )

    # Decode each sequence independently
    for seq_idx, row_idx in enumerate(valid_indices):
        new_tokens = generated_sequences[seq_idx][input_len:]
        raw_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
        results[row_idx] = _parse_response(raw_response)

    return results


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
    if getattr(args, 'limit', None):
        print(f"Limiting to first {args.limit} rows.")
        qa_data = qa_data.head(args.limit)

    # Resume from checkpoint: skip rows already written to output
    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    total = len(qa_data)
    batch_size = args.batch_size
    print(f"Running inference with batch_size={batch_size} on {total} rows.")

    rows_list = qa_data.to_dict("records")
    df_index = list(range(len(rows_list)))  # track original df positions

    for batch_start in range(0, total, batch_size):
        batch_records = rows_list[batch_start: batch_start + batch_size]
        batch_df_indices = df_index[batch_start: batch_start + batch_size]

        # Filter out already-completed rows
        pending = [
            (i, rec) for i, rec in zip(batch_df_indices, batch_records)
            if get_row_id(rec["Assigned ID"], rec["Question"]) not in completed_ids
        ]
        if not pending:
            continue

        pending_df_indices, pending_records = zip(*pending)
        batch_end = min(batch_start + batch_size, total)
        print(f"Processing rows {batch_start + 1}–{batch_end}/{total} "
              f"(batch of {len(pending_records)})...")

        responses = run_batch(model, processor, list(pending_records), args.image_dir, args.image_filename)

        # Write checkpoint immediately after each batch
        batch_df = qa_data.iloc[list(pending_df_indices)].copy()
        save_checkpoint(
            args.output_path,
            batch_df,
            {
                "predicted_answer":  [r["answer"]    for r in responses],
                "MedGemma_Reasoning": [r["reasoning"] for r in responses],
            },
        )
        completed_ids.update(get_row_id(rec["Assigned ID"], rec["Question"]) for rec in pending_records)
        print(f"  → Checkpoint saved ({len(completed_ids)}/{total} total done).")

    print(f"All done. Results written to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma Single-Slice Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, default=_cfg.get("output_base", "") + "/MedGemma1.5/single_slice_results.csv")
    parser.add_argument('--image_dir', type=str, default=_cfg.get("slice_dir"))
    parser.add_argument('--image_filename', type=str, default="Axial.png",
                        help="Filename inside each patient dir. Use 'axial_slices_montage.png' for montage.")
    parser.add_argument('--model_path', type=str, default=_cfg.get("medgemma_model_path"))
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Number of QA rows to process per model.generate() call.")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows for testing")

    args = parser.parse_args()
    main(args)