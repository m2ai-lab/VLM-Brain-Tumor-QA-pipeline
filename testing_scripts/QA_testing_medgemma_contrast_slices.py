"""
QA_testing_medgemma_contrast_slices.py — MedGemma VQA using per-contrast axial slices.

Variant of QA_testing_medgemma_multi_slice.py that loads the individually-labelled
axial PNG files produced by data_pipeline/extract_contrast_slices.py:

    <image_dir>/<pdgm_id>/axial_FLAIR.png
    <image_dir>/<pdgm_id>/axial_T1.png
    <image_dir>/<pdgm_id>/axial_T1c.png
    <image_dir>/<pdgm_id>/axial_T2.png
    ...

Rather than loading a fixed [Axial, Coronal, Sagittal] triplet, this script
discovers all axial_*.png files (typically 23 sequences including T1/T2/FLAIR, 
bias-corrected versions, DTI eddy metrics, and segmentations) and feeds them
all to MedGemma as separate image tokens. Each image is captioned in the
prompt with its contrast label (e.g. "FLAIR", "DTI_eddy_FA") so the model 
knows what it is looking at.

The prompt, JSON parsing, and Pydantic validation are identical to the
existing multi_slice script.
"""
import os
import sys
import re
import argparse

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pydantic import BaseModel, Field, ValidationError

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id

_cfg = load_config()

# ── Sequence display order (expanded to 23 requested sequences) ───────────────
_SEQUENCE_ORDER = [
    "T1", "T1_bias", "T1c", "T1C", "T1CE", "T1c_bias", 
    "T2", "T2_bias", "FLAIR", "FLAIR_bias", 
    "SWI", "SWI_bias", "ADC", "DWI", "DWI_bias", "ASL",
    "DTI_eddy_noreg", "DTI_eddy_FA", "DTI_eddy_MD", 
    "DTI_eddy_L1", "DTI_eddy_L2", "DTI_eddy_L3",
    "brain_segmentation", "brain_parenchyma_segmentation", "tumor_segmentation"
]

def _sort_key(label: str) -> tuple[int, str]:
    """
    Returns a sorting key for the image label.
    Prioritizes exact matches in _SEQUENCE_ORDER, then falls back to substring.
    """
    upper = label.upper()
    
    # 1. Try exact match first (case-insensitive)
    for i, s in enumerate(_SEQUENCE_ORDER):
        if s.upper() == upper:
            return (i, label)
            
    # 2. Try substring match (case-insensitive)
    # We check longer sequence names first to avoid "T1" matching "T1c"
    sorted_order = sorted(enumerate(_SEQUENCE_ORDER), key=lambda x: len(x[1]), reverse=True)
    for i, s in sorted_order:
        if s.upper() in upper:
            return (i, label)
            
    return (len(_SEQUENCE_ORDER), label)


# ── Few-shot example (same as multi_slice) ───────────────────────────────────
FEW_SHOT_EXAMPLE = """
Example Request:
Question: Based on the T2/FLAIR hyperintensity, what is the most likely grade? 1) Low Grade 2) High Grade

Example Response:
{
  "reasoning": "The slices show significant mass effect and central necrosis within the T1-contrast enhancing lesion, which is highly suggestive of aggressive growth.",
  "answer": "2) High Grade"
}
"""


# ── JSON cleaning / Pydantic schema (identical to multi_slice) ────────────────
def clean_json_string(raw_str: str) -> str:
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str


class MedResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step clinical observation of the MRI slices.")
    answer: str    = Field(description="The final choice selected from the options.")


# ── Image discovery ───────────────────────────────────────────────────────────

def load_contrast_slices(patient_dir: str) -> list[tuple[str, Image.Image]]:
    """
    Discover and load all axial_<CONTRAST>.png files in the patient directory.

    Returns a list of (contrast_label, PIL Image) tuples sorted by the
    standard sequence order (T1, T1c, T2, FLAIR, ADC, …).
    """
    if not os.path.isdir(patient_dir):
        return []

    pairs = []
    for fname in os.listdir(patient_dir):
        if not (fname.startswith("axial_") and fname.endswith(".png")):
            continue
        label = fname[len("axial_"):-len(".png")]  # strip prefix/suffix
        fpath = os.path.join(patient_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            pairs.append((label, img))
        except Exception as exc:
            print(f"    [WARN] Could not load {fpath}: {exc}")

    # Sort by clinical sequence order
    pairs.sort(key=lambda t: _sort_key(t[0]))
    return pairs


# ── Inference ─────────────────────────────────────────────────────────────────

def run_batch(model, processor, batch_rows: list[dict], base_image_dir: str) -> list[dict]:
    """
    Run inference on a batch of QA rows, each using multiple contrast slices.
    """
    valid_indices: list[int] = []
    results: list[dict] = [{"reasoning": "Not processed", "answer": "Error"}] * len(batch_rows)

    texts: list[str] = []
    images_per_patient: list[list[Image.Image]] = []

    for i, row in enumerate(batch_rows):
        patient_id = row["Assigned ID"]
        patient_dir = os.path.join(base_image_dir, str(patient_id))
        contrast_pairs = load_contrast_slices(patient_dir)

        if not contrast_pairs:
            results[i] = {
                "reasoning": f"Error: No axial_*.png files found in {patient_dir}",
                "answer": "Error",
            }
            continue

        labels = [label for label, _ in contrast_pairs]
        images = [img   for _, img   in contrast_pairs]

        img_descriptions = " | ".join(f"Image {i+1}: {lbl}" for i, lbl in enumerate(labels))
        prompt_text = (
            "Instruction: You are a neuroradiologist. Analyze the provided MRI "
            f"contrast sequences and provide a structured JSON response.\n"
            f"Available contrasts: {img_descriptions}\n"
            f"{FEW_SHOT_EXAMPLE}"
            "---\n"
            f"Actual Question: {row['Question']}\n"
            "Response:"
        )

        content = [{"type": "image"}] * len(images)
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        valid_indices.append(i)
        texts.append(input_text)
        images_per_patient.append(images)

    if not valid_indices:
        return results

    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=texts,
        images=images_per_patient,
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

    for seq_idx, row_idx in enumerate(valid_indices):
        new_tokens = generated_sequences[seq_idx][input_len:]
        raw_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
        results[row_idx] = _parse_response("{" + raw_response)  # Ensure opening brace for parsing if missing

    return results


def _parse_response(raw_response: str) -> dict:
    cleaned = clean_json_string(raw_response)
    try:
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model output")
        validated = MedResponse.model_validate_json(json_match.group(0))
        return validated.model_dump()
    except Exception as e:
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"Loading MedGemma from {args.model_path}…")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    qa_data = pd.read_csv(args.qa_path)
    if getattr(args, 'limit', None):
        print(f"Limiting to first {args.limit} rows.")
        qa_data = qa_data.head(args.limit)

    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    total = len(qa_data)
    batch_size = args.batch_size
    print(f"Running contrast-slices inference (batch_size={batch_size}) on {total} rows.")

    rows_list = qa_data.to_dict("records")
    df_index = list(range(len(rows_list)))

    for batch_start in range(0, total, batch_size):
        batch_records = rows_list[batch_start: batch_start + batch_size]
        batch_df_indices = df_index[batch_start: batch_start + batch_size]

        pending = [
            (i, rec) for i, rec in zip(batch_df_indices, batch_records)
            if get_row_id(rec["Assigned ID"], rec["Question"]) not in completed_ids
        ]
        if not pending:
            continue

        pending_df_indices, pending_records = zip(*pending)
        batch_end = min(batch_start + batch_size, total)
        print(f"Processing rows {batch_start + 1}–{batch_end}/{total} (batch of {len(pending_records)})…")

        responses = run_batch(model, processor, list(pending_records), args.image_dir)

        batch_df = qa_data.iloc[list(pending_df_indices)].copy()
        save_checkpoint(
            args.output_path,
            batch_df,
            {
                "predicted_answer":   [r["answer"] for r in responses],
                "MedGemma_Reasoning": [r["reasoning"] for r in responses],
            },
        )
        completed_ids.update(get_row_id(rec["Assigned ID"], rec["Question"]) for rec in pending_records)
        print(f"  → Checkpoint saved ({len(completed_ids)}/{total} total done).")

    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedGemma VQA using per-contrast axial slices (axial_*.png)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qa_path", type=str, default=_cfg.get("qa_path"))
    parser.add_argument(
        "--output_path", type=str,
        default=_cfg.get("output_base", "") + "/MedGemma1.5/contrast_slices_results.csv",
    )
    parser.add_argument(
        "--image_dir", type=str, default=_cfg.get("slice_dir"),
        help="Root dir containing <pdgm_id>/axial_*.png files.",
    )
    parser.add_argument("--model_path", type=str, default=_cfg.get("medgemma_model_path"))
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of patients to process in parallel. (Ensure GPU memory can handle batch_size * 24 images).",
    )
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows for testing")

    args = parser.parse_args()
    main(args)
