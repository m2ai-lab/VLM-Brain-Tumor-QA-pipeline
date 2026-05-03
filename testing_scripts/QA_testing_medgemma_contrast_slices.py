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
discovers however many axial_*.png files exist for the patient and feeds them
all to MedGemma as separate image tokens.  Each image is captioned in the
prompt with its contrast label (e.g. "FLAIR", "T1c") so the model knows what
it is looking at.

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
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint

_cfg = load_config()

# ── Sequence display order (mirrors montage_slices.py) ───────────────────────
_SEQUENCE_ORDER = ["T1", "T1c", "T1C", "T1CE", "T2", "FLAIR", "ADC", "DWI", "SWI"]

def _sort_key(label: str) -> tuple[int, str]:
    upper = label.upper()
    for i, s in enumerate(_SEQUENCE_ORDER):
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

def query_the_model(model, processor, question: str,
                    patient_id: str, base_image_dir: str) -> dict:
    """
    Run MedGemma on all per-contrast axial slices for one patient.

    Each contrast PNG becomes its own image token; the prompt lists the
    contrast names in order so the model has sequence context.
    """
    patient_dir = os.path.join(base_image_dir, str(patient_id))
    contrast_pairs = load_contrast_slices(patient_dir)

    if not contrast_pairs:
        return {
            "reasoning": f"Error: No axial_*.png files found in {patient_dir}",
            "answer": "Error",
        }

    labels = [label for label, _ in contrast_pairs]
    images = [img   for _, img   in contrast_pairs]

    # Build a contrast list description for the prompt
    # e.g. "Image 1: T1 | Image 2: T1c | Image 3: T2 | Image 4: FLAIR"
    img_descriptions = " | ".join(
        f"Image {i+1}: {lbl}" for i, lbl in enumerate(labels)
    )

    prompt_text = (
        "Instruction: You are a neuroradiologist. Analyze the provided MRI "
        f"contrast sequences and provide a structured JSON response.\n"
        f"Available contrasts: {img_descriptions}\n"
        f"{FEW_SHOT_EXAMPLE}"
        "---\n"
        f"Actual Question: {question}\n"
        "Response:"
    )

    # One {"type": "image"} entry per contrast, then the text
    content = [{"type": "image"}] * len(images)
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Gemma3 processor requires images as a nested list even for a single sample:
    # [images] = "1 text sample whose prompt has len(images) <image> tokens".
    # Passing a flat list causes a batch-size mismatch error.
    inputs = processor(
        text=input_text,
        images=[images],
        padding=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_sequence = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            stop_strings=["}"],
            tokenizer=processor.tokenizer,
        )

    new_tokens    = generated_sequence[0][input_len:]
    raw_response  = processor.decode(new_tokens, skip_special_tokens=True).strip()
    full_json_str = "{" + raw_response
    if not full_json_str.endswith("}"):
        full_json_str += "}"

    cleaned = clean_json_string(raw_response)

    try:
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model output")
        validated = MedResponse.model_validate_json(json_match.group(0))
        return validated.model_dump()
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": cleaned}
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
    total = qa_data.shape[0]

    # Contrast-slices variant keeps batch_size=1 because each patient has a
    # variable number of images, making true multi-patient batching non-trivial.
    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    print(f"Running contrast-slices inference (batch_size=1) on {total} rows.")

    for idx, row in qa_data.iterrows():
        patient_id = row["Assigned ID"]
        if str(patient_id) in completed_ids:
            continue

        print(f"Processing {idx+1}/{total} (ID: {patient_id})…")

        # Log which contrasts will be fed in
        patient_dir = os.path.join(args.image_dir, str(patient_id))
        pairs = load_contrast_slices(patient_dir)
        if pairs:
            print(f"  Contrasts: {[lbl for lbl, _ in pairs]}")

        response = query_the_model(
            model, processor, row["Question"], patient_id, args.image_dir
        )
        print(f"  Response: {response['answer']}\n{'-'*30}")

        # Write checkpoint immediately after each row
        save_checkpoint(
            args.output_path,
            qa_data.iloc[[idx]],
            {
                "predicted_answer":   [response["answer"]],
                "MedGemma_Reasoning": [response["reasoning"]],
            },
        )
        completed_ids.add(str(patient_id))

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
        help="Fixed at 1 for contrast_slices: variable image count per patient prevents true batching.",
    )

    args = parser.parse_args()
    main(args)
