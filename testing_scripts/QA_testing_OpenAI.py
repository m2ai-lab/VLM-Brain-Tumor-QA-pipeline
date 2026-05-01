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
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field


# ── Debug helper (defined before anything else so all startup lines can use it) ──
def _dbg(msg: str) -> None:
    """Print a timestamped debug line immediately. flush=True defeats conda output buffering."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

_dbg("Script starting — imports OK")

# ── Load credentials from .env ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_dbg(f"Project root : {_PROJECT_ROOT}")
_dbg(f"Loading .env : {_PROJECT_ROOT / '.env'}")
loaded = load_dotenv(_PROJECT_ROOT / ".env", override=True)
_dbg(f".env loaded  : {loaded}  (False = file not found)")

from config_utils import load_config
_cfg = load_config()
_dbg("config.yaml  : loaded OK")

# Variable names match openai_vision_models.ipynb cell 2
API_KEY           = os.environ.get("OPENAI_API_KEY")
API_VERSION       = os.environ.get("API_VERSION", "2025-04-01-preview")
RESOURCE_ENDPOINT = os.environ.get("RESOURCE_ENDPOINT", "https://unified-api.ucsf.edu/general")

_dbg(f"API_VERSION       : {API_VERSION}")
_dbg(f"RESOURCE_ENDPOINT : {RESOURCE_ENDPOINT}")
_dbg(f"OPENAI_API_KEY    : {'SET (' + str(len(API_KEY)) + ' chars)' if API_KEY else 'NOT SET — check .env'}")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Make sure it is set in your .env file."
    )

# ── Initialise AzureOpenAI client (notebook cell 4) ───────────────────────────
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)
_dbg("AzureOpenAI client initialised OK")


# ── Deployment (notebook cells 5 & 11) ────────────────────────────────────────
# gpt-5-mini is vision-capable and available through Versa (per notebook output).
# Other valid vision deployments: gpt-4.1-2025-04-14, gpt-4o-2024-11-20, o4-mini-2025-04-16
DEFAULT_DEPLOYMENT = "gpt-5-mini-2025-08-07"

# ── Retry settings ────────────────────────────────────────────────────────────
RETRY_SECS  = 10
MAX_RETRIES = 4

# ── Token budget for GPT-5+ and reasoning models ──────────────────────────────
# Notebook cell 11: use max_completion_tokens (not max_tokens) for o- and GPT5+ models.
MAX_COMPLETION_TOKENS = 1024

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
    """
    from PIL import Image
    import io

    file_bytes = os.path.getsize(image_path)
    _dbg(f"  Loading image : {image_path} ({file_bytes:,} bytes)")
    img = Image.open(image_path).convert("RGB")
    _dbg(f"  Image size    : {img.size[0]}x{img.size[1]} px")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    _dbg(f"  Base64 length : {len(b64):,} chars")
    return b64


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
            _dbg(f"  → Sending API request (attempt {retries + 1}/{MAX_RETRIES + 1}, deployment={deployment})…")
            t0 = time.monotonic()

            # beta.parse enforces the Pydantic schema on the model response
            response = client.beta.chat.completions.parse(
                model=deployment,
                messages=messages,
                response_format=VQAResponse,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            elapsed = time.monotonic() - t0
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            _dbg(
                f"  ← Response received in {elapsed:.1f}s | "
                f"finish={finish_reason!r} | "
                f"tokens: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"total={usage.total_tokens}"
            )

            if finish_reason not in ("stop", "length"):
                _dbg(f"  [WARN] Unexpected finish_reason: {finish_reason!r}")

            parsed: VQAResponse = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model returned null parsed content (parsed=None)")
            return parsed

        except Exception as e:
            if retries >= MAX_RETRIES:
                raise RuntimeError(
                    f"AzureOpenAI call failed after {MAX_RETRIES + 1} attempts. "
                    f"Deployment: {deployment}. Error: {e}"
                )
            _dbg(
                f"  [WARN] Attempt {retries + 1}/{MAX_RETRIES + 1} failed: {e}. "
                f"Retrying in {RETRY_SECS}s…"
            )
            retries += 1
            time.sleep(RETRY_SECS)


# ── Per-patient inference ─────────────────────────────────────────────────────

def query_the_model(
    question: str,
    patient_id: str,
    base_image_dir: str,
    deployment: str,
    image_filename: str = "Axial.png",
) -> dict:
    """
    Send one question + the patient's axial MRI slice to GPT-5+ via Versa.

    Returns {"answer": ..., "concise_reasoning": ...} from the Pydantic VQAResponse.
    Mirrors query_the_model() in QA_testing_MedImageInsight.py.
    """
    patient_image_path = os.path.join(base_image_dir, str(patient_id), image_filename)
    _dbg(f"  Image path: {patient_image_path}")

    if not os.path.exists(patient_image_path):
        _dbg(f"  [ERROR] Image not found")
        return {
            "concise_reasoning": f"Error: Image not found at {patient_image_path}",
            "answer": "Error",
        }

    try:
        image_b64 = encode_image_as_base64(patient_image_path)
    except Exception as e:
        _dbg(f"  [ERROR] Image load failed: {e}")
        return {"concise_reasoning": f"Image load error: {e}", "answer": "Error"}

    try:
        result: VQAResponse = call_versa_vision(image_b64, question, deployment)
    except RuntimeError as e:
        _dbg(f"  [ERROR] API call failed: {e}")
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

    if args.limit:
        qa_data = qa_data.head(args.limit)
        _dbg(f"[LIMIT] Running on first {args.limit} rows only (smoke test).")

    generated_answer    = []
    generated_reasoning = []
    total = qa_data.shape[0]

    _dbg(f"Loaded {total} rows from {args.qa_path}")
    _dbg(f"Deployment      : {args.deployment}")
    _dbg(f"Endpoint        : {RESOURCE_ENDPOINT}")
    _dbg(f"API version     : {API_VERSION}")
    _dbg(f"Max tokens      : {MAX_COMPLETION_TOKENS}")
    _dbg(f"Image dir       : {args.image_dir or '(blank mode)'}")
    _dbg(f"Blank image     : {args.image_path or '(none)'}")
    _dbg(f"Output path     : {args.output_path}")
    print("-" * 60, flush=True)

    for idx, row in qa_data.iterrows():
        _dbg(f"Row {idx + 1}/{total} | Patient: {row['Assigned ID']}")

        if args.image_path:
            response = query_the_model_blank(
                question   = row["Question"],
                patient_id = row["Assigned ID"],
                image_path = args.image_path,
                deployment = args.deployment,
            )
        else:
            response = query_the_model(
                question       = row["Question"],
                patient_id     = row["Assigned ID"],
                base_image_dir = args.image_dir,
                deployment     = args.deployment,
                image_filename = args.image_filename,
            )

        generated_answer.append(response["answer"])
        generated_reasoning.append(response["concise_reasoning"])
        _dbg(f"  ✓ answer: {response['answer']}")
        _dbg(f"    reasoning: {response['concise_reasoning']}")
        print("-" * 30, flush=True)

    _dbg(f"All {total} rows processed. Saving to {args.output_path}…")
    qa_data["predicted_answer"]      = generated_answer
    qa_data["GPT_Concise_Reasoning"] = generated_reasoning

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    qa_data.to_csv(args.output_path, index=False)
    _dbg(f"Done. Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-5+ vision VQA via UCSF Versa/Mulesoft Azure OpenAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_path", type=str,
        default=_cfg.get("qa_path"),
        help="QA CSV with 'Assigned ID' and 'Question' columns.",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=os.path.join(_cfg.get("output_base", "test_output"), "OpenAI", "single_slice_results.csv"),
        help="Path for the output results CSV.",
    )
    parser.add_argument(
        "--image_dir", type=str,
        default=_cfg.get("slice_dir"),
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
    parser.add_argument(
        "--image_filename", type=str, default="Axial.png",
        help="Filename of the image inside each patient's directory. "
             "Use 'axial_slices_montage.png' for the montage variant.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N rows. Useful for quick local smoke tests.",
    )

    main(parser.parse_args())