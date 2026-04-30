"""
QA_testing_OpenAI.py — VQA using GPT-5+ vision via UCSF Versa / Mulesoft Azure OpenAI API.

Mirrors QA_testing_MedImageInsight.py in pipeline structure and uses the
AzureOpenAI SDK pattern from openai_vision_models.ipynb.

  1. Initialise an AzureOpenAI client (same as notebook cell 4)
  2. For each QA row, load and base64-encode the patient's MRI slice (same as notebook cell 17)
  3. Build a vision message (text + image_url, same layout as notebook cells 17 & 20)
  4. Call client.beta.chat.completions.parse with a Pydantic VQAResponse schema
     so the model always returns structured JSON — no regex heuristics needed
  5. Validate finish_reason == 'stop' (notebook cell 7)
  6. Write a results CSV with predicted_answer and GPT_Concise_Reasoning columns

Key SDK choices (from openai_vision_models.ipynb):
  - AzureOpenAI client (notebook cell 4) instead of raw requests
  - API version 2025-04-01-preview — required for GPT-5+ vision (notebook cell 2)
  - max_completion_tokens (not max_tokens) for GPT-5+/reasoning models (notebook cell 11)
  - OPENAI_API_KEY env var name (notebook cell 2)
  - client.beta.chat.completions.parse for Pydantic structured output

Credentials are loaded from the project .env file:
  OPENAI_API_KEY     — base64 Mulesoft composite key (88 chars)
  API_VERSION        — 2025-04-01-preview
  RESOURCE_ENDPOINT  — https://unified-api.ucsf.edu/general  (no trailing slash)
"""
import os
import re
import base64
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

# ── Load credentials from .env ────────────────────────────────────────────────
# Resolve .env relative to the project root (one level up from this script),
# mirroring the find_dotenv() pattern used in openai_vision_models.ipynb.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# Variable names match openai_vision_models.ipynb cell 2
API_KEY           = os.environ.get("OPENAI_API_KEY")
API_VERSION       = os.environ.get("API_VERSION", "2025-04-01-preview")
RESOURCE_ENDPOINT = os.environ.get("RESOURCE_ENDPOINT", "https://unified-api.ucsf.edu/general")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Make sure it is set in your .env file."
    )

# ── Initialise AzureOpenAI client (notebook cell 4) ───────────────────────────
# This is the recommended SDK approach from openai_vision_models.ipynb.
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

# ── Deployment (notebook cells 5 & 11) ────────────────────────────────────────
# gpt-5-mini is vision-capable and available through Versa (per notebook output).
# Other valid vision deployments: gpt-4.1-2025-04-14, gpt-4o-2024-11-20, o4-mini-2025-04-16
DEFAULT_DEPLOYMENT = "gpt-5-mini-2025-08-07"

# ── Retry settings ────────────────────────────────────────────────────────────
RETRY_SECS  = 10
MAX_RETRIES = 4

# ── Token budget for GPT-5+ and reasoning models ──────────────────────────────
# Notebook cell 11: use max_completion_tokens (not max_tokens) for o- and GPT5+ models.
MAX_COMPLETION_TOKENS = 256

# ── Pydantic response schema ──────────────────────────────────────────────────
class VQAResponse(BaseModel):
    """Structured output for one brain MRI VQA question."""
    answer: str = Field(
        description="The chosen answer option verbatim, e.g. '1) Low Grade'."
    )
    concise_reasoning: str = Field(
        description=(
            "One sentence (≤25 words) citing the specific MRI finding that "
            "supports the chosen answer."
        )
    )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a radiologist reviewing brain MRI scans. "
    "Answer the multiple-choice question based on the provided MRI image. "
    "Your response must be a JSON object with exactly two fields:\n"
    "  answer           — the chosen option verbatim (e.g. '1) Low Grade')\n"
    "  concise_reasoning — one sentence (≤25 words) citing the MRI finding "
    "that supports your answer."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_answer_options(question: str) -> tuple[str, list[str]]:
    """
    Split a 4-option multiple-choice question into its stem and options.

    Uses the same proven regex as the rest of the pipeline
    (dataset_reshuffle.py, QA_testing_MedImageInsight.py).

    Returns:
        (question_stem, ["1) Option A", "2) Option B", "3) Option C", "4) Option D"])
    """
    parts = re.split(r"([0-5]\))", question)
    stem  = parts[0].strip()

    options = []
    for i in range(1, len(parts) - 1, 2):
        marker = parts[i]
        text   = parts[i + 1].strip()
        options.append(f"{marker} {text}")

    if len(options) != 4:
        raise ValueError(
            f"Expected 4 answer options, got {len(options)} from: {question[:120]}..."
        )
    return stem, options


def encode_image_as_base64(image_path: str) -> str:
    """
    Read an image file, force RGB, and return its base64-encoded PNG string.

    Mirrors the encode_image() helper in openai_vision_models.ipynb (cell 17),
    with an added PIL force-RGB step to handle RGBA/grayscale MRI exports.
    """
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_versa_vision(image_b64: str, question: str, deployment: str) -> VQAResponse:
    """
    Send a vision message to GPT-5+ via the AzureOpenAI SDK and return a
    parsed VQAResponse (Pydantic structured output).

    Uses client.beta.chat.completions.parse with response_format=VQAResponse so
    the model is constrained to return valid JSON matching the schema — no
    regex heuristics needed to extract the answer.

    Message structure mirrors openai_vision_models.ipynb cells 17 & 20:
      - system role with radiology prompt
      - user role with [text question, image_url (base64 data URI)]

    Uses max_completion_tokens as required by GPT-5+/reasoning models (cell 11).
    Validates finish_reason == 'stop' (cell 7).
    Retries on transient failures.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        # base64 data URI — same format as notebook cell 17
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        },
    ]

    retries = 0
    while True:
        try:
            # beta.parse enforces the Pydantic schema on the model response
            response = client.beta.chat.completions.parse(
                model=deployment,
                messages=messages,
                response_format=VQAResponse,
                max_completion_tokens=MAX_COMPLETION_TOKENS,  # GPT-5+ requirement (cell 11)
            )

            # Validate finish_reason (notebook cell 7)
            finish_reason = response.choices[0].finish_reason
            if finish_reason not in ("stop", "length"):
                print(f"  [WARN] Unexpected finish_reason: {finish_reason!r}")

            parsed: VQAResponse = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model returned null parsed content")
            return parsed

        except Exception as e:
            if retries >= MAX_RETRIES:
                raise RuntimeError(
                    f"AzureOpenAI call failed after {MAX_RETRIES + 1} attempts. "
                    f"Deployment: {deployment}. Error: {e}"
                )
            print(
                f"  [WARN] Attempt {retries + 1}/{MAX_RETRIES + 1} failed "
                f"({e}). Retrying in {RETRY_SECS}s…"
            )
            retries += 1
            time.sleep(RETRY_SECS)


# ── Per-patient inference ─────────────────────────────────────────────────────

def query_the_model(
    question: str,
    patient_id: str,
    base_image_dir: str,
    deployment: str,
) -> dict:
    """
    Send one question + the patient's axial MRI slice to GPT-5+ via Versa.

    Returns {"answer": ..., "concise_reasoning": ...} from the Pydantic VQAResponse.
    Mirrors query_the_model() in QA_testing_MedImageInsight.py.
    """
    patient_image_path = os.path.join(base_image_dir, str(patient_id), "Axial.png")

    if not os.path.exists(patient_image_path):
        return {
            "concise_reasoning": f"Error: Image not found at {patient_image_path}",
            "answer": "Error",
        }

    try:
        image_b64 = encode_image_as_base64(patient_image_path)
    except Exception as e:
        return {"concise_reasoning": f"Image load error: {e}", "answer": "Error"}

    try:
        result: VQAResponse = call_versa_vision(image_b64, question, deployment)
    except RuntimeError as e:
        return {"concise_reasoning": str(e), "answer": "Error"}

    return {"answer": result.answer, "concise_reasoning": result.concise_reasoning}


def query_the_model_blank(
    question: str,
    patient_id: str,
    image_path: str,
    deployment: str,
) -> dict:
    """
    Same as query_the_model but uses a single fixed blank image for all patients
    (control / blank experiment).

    Mirrors query_the_model_blank() in QA_testing_MedImageInsight.py.
    """
    if not os.path.exists(image_path):
        return {
            "concise_reasoning": f"Error: Blank image not found at {image_path}",
            "answer": "Error",
        }

    try:
        image_b64 = encode_image_as_base64(image_path)
    except Exception as e:
        return {"concise_reasoning": f"Image load error: {e}", "answer": "Error"}

    try:
        result: VQAResponse = call_versa_vision(image_b64, question, deployment)
    except RuntimeError as e:
        return {"concise_reasoning": str(e), "answer": "Error"}

    return {"answer": result.answer, "concise_reasoning": result.concise_reasoning}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    qa_data = pd.read_csv(args.qa_path)
    generated_answer    = []
    generated_reasoning = []
    total = qa_data.shape[0]

    print(f"Loaded {total} rows from {args.qa_path}")
    print(f"Using deployment       : {args.deployment}")
    print(f"Versa endpoint         : {RESOURCE_ENDPOINT}")
    print(f"API version            : {API_VERSION}")
    print(f"max_completion_tokens  : {MAX_COMPLETION_TOKENS}")
    print(f"Output schema          : VQAResponse (answer + concise_reasoning)")
    print("-" * 60)

    for idx, row in qa_data.iterrows():
        print(f"Processing {idx + 1}/{total} (ID: {row['Assigned ID']})…")

        if args.image_path:
            # Blank / control experiment — same blacked-out image for every row
            response = query_the_model_blank(
                question   = row["Question"],
                patient_id = row["Assigned ID"],
                image_path = args.image_path,
                deployment = args.deployment,
            )
        else:
            # Normal experiment — per-patient image directory
            response = query_the_model(
                question       = row["Question"],
                patient_id     = row["Assigned ID"],
                base_image_dir = args.image_dir,
                deployment     = args.deployment,
            )

        generated_answer.append(response["answer"])
        generated_reasoning.append(response["concise_reasoning"])
        print(f"  → {response['answer']} | {response['concise_reasoning']}")
        print("-" * 30)

    # Save results (same columns as the rest of the pipeline)
    qa_data["predicted_answer"]      = generated_answer
    qa_data["GPT_Concise_Reasoning"] = generated_reasoning

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    qa_data.to_csv(args.output_path, index=False)
    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-5+ vision VQA via UCSF Versa/Mulesoft Azure OpenAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_path", type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv",
        help="QA CSV with 'Assigned ID' and 'Question' columns.",
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/OpenAI/single_slice_results.csv",
        help="Path for the output results CSV.",
    )
    parser.add_argument(
        "--image_dir", type=str,
        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices",
        help="Root directory of per-patient Axial.png images.",
    )
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="For blank experiments: path to a single blacked-out image. "
             "Overrides --image_dir.",
    )
    parser.add_argument(
        "--deployment", type=str, default=DEFAULT_DEPLOYMENT,
        help=(
            "Versa Azure deployment name (must be vision-capable). "
            "Options: gpt-5-mini-2025-08-07, gpt-4.1-2025-04-14, "
            "gpt-4o-2024-11-20, o4-mini-2025-04-16"
        ),
    )

    main(parser.parse_args())