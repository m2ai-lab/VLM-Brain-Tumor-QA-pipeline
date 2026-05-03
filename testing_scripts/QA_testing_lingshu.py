"""
QA_testing_lingshu.py — Batched inference for Lingshu-32B (Qwen2.5-VL based) on Brain MRI VQA.

Handles single_slice, montage_slice, and blank variants.
Uses Pydantic + JSON-constrained generation (via prompt) to ensure consistent output.
"""

import os
import pandas as pd
import torch
import argparse
import json
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint

_cfg = load_config()

# --- Pydantic Schema ---
class VQAResponse(BaseModel):
    concise_reasoning: str = Field(description="Brief clinical reasoning for the choice.")
    answer: str = Field(description="The final choice selected from the options.")

FEW_SHOT_EXAMPLE = """
Example:
Question: Is there evidence of a midline shift?
1) Yes
2) No
3) Indeterminate
4) Not applicable

Response:
{
  "concise_reasoning": "The septum pellucidum is at the midline with no displacement of ventricular structures.",
  "answer": "2) No"
}
"""


def _build_prompt(question: str) -> str:
    return (
        "Instruction: You are a neuroradiologist. Analyze the MRI slice and provide a structured JSON response.\n"
        "CRITICAL: Your 'answer' field MUST contain the EXACT verbatim answer text from the options provided, "
        "including the number/letter marker (e.g., '1) Left frontal lobe'). Do NOT just output the number.\n"
        f"{FEW_SHOT_EXAMPLE}\n"
        "---\n"
        f"Question: {question}\n"
        "Return ONLY the JSON object."
    )


def _parse_response(output_text: str) -> dict:
    try:
        if "{" in output_text and "}" in output_text:
            json_str = output_text[output_text.find("{"):output_text.rfind("}")+1]
            data = json.loads(json_str)
        else:
            data = json.loads(output_text)
        return {
            "concise_reasoning": data.get("concise_reasoning", "No reasoning provided."),
            "answer": data.get("answer", "Unknown"),
        }
    except Exception as e:
        return {
            "concise_reasoning": f"JSON Parse Error: {e}. Raw: {output_text}",
            "answer": "Error",
        }


def run_batch(
    model,
    processor,
    batch_rows: list[dict],
    base_image_dir: str,
    image_filename: str,
    image_path_override: str | None = None,
) -> list[dict]:
    """
    Run batched inference on Lingshu-32B for a list of QA rows.

    For blank variants, pass image_path_override to use a single fixed image.
    Returns one response dict per row in the same order.
    """
    valid_indices: list[int] = []
    results: list[dict] = [{"concise_reasoning": "Not processed", "answer": "Error"}] * len(batch_rows)

    all_messages: list[list[dict]] = []

    for i, row in enumerate(batch_rows):
        if image_path_override:
            patient_image_path = image_path_override
        else:
            patient_image_path = os.path.join(base_image_dir, str(row["Assigned ID"]), image_filename)

        if not os.path.exists(patient_image_path):
            results[i] = {
                "concise_reasoning": f"Error: Image not found at {patient_image_path}",
                "answer": "Error",
            }
            continue

        prompt_text = _build_prompt(row["Question"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": patient_image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        valid_indices.append(i)
        all_messages.append(messages)

    if not valid_indices:
        return results

    # Build batched text + image inputs
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # process_vision_info flattens the image list across all messages in the batch
    image_inputs, video_inputs = process_vision_info(
        [msg for msgs in all_messages for msg in msgs]
    )

    # Left-padding required for batched decoder-only generation
    processor.tokenizer.padding_side = "left"

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for seq_idx, row_idx in enumerate(valid_indices):
        results[row_idx] = _parse_response(output_texts[seq_idx])

    return results


def main(args):
    print(f"Loading model: {args.model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading QA CSV from: {args.qa_path}")
    qa_data = pd.read_csv(args.qa_path)

    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        qa_data = qa_data.head(args.limit)

    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    total = len(qa_data)
    batch_size = args.batch_size
    print(f"Running inference with batch_size={batch_size} on {total} rows.")

    rows_list = qa_data.to_dict("records")
    df_index = list(range(len(rows_list)))

    for batch_start in range(0, total, batch_size):
        batch_records = rows_list[batch_start: batch_start + batch_size]
        batch_df_indices = df_index[batch_start: batch_start + batch_size]

        pending = [
            (i, rec) for i, rec in zip(batch_df_indices, batch_records)
            if str(rec["Assigned ID"]) not in completed_ids
        ]
        if not pending:
            continue

        pending_df_indices, pending_records = zip(*pending)
        batch_end = min(batch_start + batch_size, total)
        print(f"Processing rows {batch_start + 1}–{batch_end}/{total} "
              f"(batch of {len(pending_records)})...")

        responses = run_batch(
            model, processor,
            list(pending_records),
            args.image_dir,
            image_filename=args.image_filename,
            image_path_override=args.image_path,
        )

        batch_df = qa_data.iloc[list(pending_df_indices)].copy()
        save_checkpoint(
            args.output_path,
            batch_df,
            {
                "predicted_answer":  [r["answer"]             for r in responses],
                "Lingshu_Reasoning": [r["concise_reasoning"]  for r in responses],
            },
        )
        completed_ids.update(str(rec["Assigned ID"]) for rec in pending_records)
        print(f"  → Checkpoint saved ({len(completed_ids)}/{total} total done).")

    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu-32B VQA Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=_cfg.get("lingshu_model_path", "models/Lingshu-32B"))
    parser.add_argument('--image_dir', type=str, default=_cfg.get("slice_dir"))
    parser.add_argument('--image_path', type=str, default=None, help="Path to single image (e.g. for blank variant)")
    parser.add_argument('--image_filename', type=str, default="Axial.png")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Number of QA rows per model.generate() call. Keep low (2) for 32B model.")
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()
    main(args)
