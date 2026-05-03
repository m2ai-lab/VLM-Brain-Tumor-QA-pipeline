"""
QA_testing_OpenAI.py — VQA using GPT-5+ vision via UCSF Versa / Mulesoft Azure OpenAI API.

Features:
  - Parallel Request Batching: Uses threads to send multiple requests at once.
  - Checkpointing: Resumes from where it left off and saves progress row-by-row.
  - Structured Output: Uses Pydantic schemas for reliable JSON responses.
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id


# ── Debug helper ──────────────────────────────────────────────────────────────
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

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Make sure it is set in your .env file.")

# ── Initialise AzureOpenAI client ─────────────────────────────────────────────
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)
_dbg("AzureOpenAI client initialised OK")

# ── Global Settings & Threading ───────────────────────────────────────────────
DEFAULT_DEPLOYMENT = "gpt-5-mini-2025-08-07"
RETRY_SECS  = 10
MAX_RETRIES = 4
MAX_COMPLETION_TOKENS = 1024

_SAVE_LOCK = threading.Lock()  # Protects CSV appends from parallel threads

# ── Pydantic response schema ──────────────────────────────────────────────────
class VQAResponse(BaseModel):
    """Structured output for one brain MRI VQA question."""
    answer: str = Field(description="The chosen answer option verbatim, e.g. '1) Low Grade'.")
    concise_reasoning: str = Field(description="One sentence citing the MRI finding.")

SYSTEM_PROMPT = (
    "You are a radiologist reviewing brain MRI scans. "
    "Answer the multiple-choice question based on the provided MRI image. "
    "Your response must be a JSON object with 'answer' and 'concise_reasoning'."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image_as_base64(image_path: str) -> str:
    from PIL import Image
    import io
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_versa_vision(image_b64: str, question: str, deployment: str) -> VQAResponse:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]
    retries = 0
    while True:
        try:
            response = client.beta.chat.completions.parse(
                model=deployment,
                messages=messages,
                response_format=VQAResponse,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            parsed: VQAResponse = response.choices[0].message.parsed
            if parsed is None: raise ValueError("Model returned null parsed content")
            return parsed
        except Exception as e:
            if retries >= MAX_RETRIES: raise RuntimeError(f"API failed after {MAX_RETRIES+1} attempts: {e}")
            time.sleep(RETRY_SECS)
            retries += 1

# ── Parallel Task Worker ──────────────────────────────────────────────────────

def process_row(idx: int, row: pd.Series, args: argparse.Namespace) -> dict:
    """Task function for ThreadPoolExecutor."""
    try:
        if args.image_path:
            # Control / Blank mode
            img_path = args.image_path
        else:
            # Normal mode
            img_path = os.path.join(args.image_dir, str(row["Assigned ID"]), args.image_filename)

        if not os.path.exists(img_path):
            return {"rid": get_row_id(row["Assigned ID"], row["Question"]), "answer": "Error", "reasoning": f"Image not found: {img_path}", "idx": idx}

        image_b64 = encode_image_as_base64(img_path)
        result = call_versa_vision(image_b64, row["Question"], args.deployment)
        
        return {
            "rid": get_row_id(row["Assigned ID"], row["Question"]),
            "answer": result.answer,
            "reasoning": result.concise_reasoning,
            "idx": idx,
            "row_data": row
        }
    except Exception as e:
        _dbg(f"  [ERROR] Row {idx} failed: {e}")
        return {"rid": get_row_id(row["Assigned ID"], row["Question"]), "answer": "Error", "reasoning": str(e), "idx": idx}

# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # 1. Load progress
    processed_ids = load_checkpoint(args.output_path)
    qa_data = pd.read_csv(args.qa_path)
    if args.limit:
        qa_data = qa_data.head(args.limit)

    to_process = []
    for idx, row in qa_data.iterrows():
        if get_row_id(row["Assigned ID"], row["Question"]) not in processed_ids:
            to_process.append((idx, row))

    if not to_process:
        _dbg("All rows processed.")
        return

    _dbg(f"Loaded {len(qa_data)} rows. Resuming from {len(processed_ids)} existing results.")
    _dbg(f"Processing {len(to_process)} remaining rows using {args.batch_size} parallel workers...")

    # 2. Parallel execution
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(process_row, idx, row, args): idx for idx, row in to_process}
        
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            
            # 3. Thread-safe save
            with _SAVE_LOCK:
                row_slice = qa_data.iloc[[res["idx"]]]
                save_checkpoint(
                    args.output_path, 
                    row_slice, 
                    {"predicted_answer": [res["answer"]], "GPT_Concise_Reasoning": [res["reasoning"]]}
                )
            
            done_count += 1
            if done_count % 5 == 0 or done_count == len(to_process):
                _dbg(f"Progress: {done_count}/{len(to_process)} requests completed.")

    _dbg(f"Run complete. Results in {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel GPT-5+ vision VQA via Versa")
    parser.add_argument("--qa_path", type=str, default=_cfg.get("qa_path"))
    parser.add_argument("--output_path", type=str, 
                        default=os.path.join(_cfg.get("output_base", "test_output"), "OpenAI", "results.csv"))
    parser.add_argument("--image_dir", type=str, default=_cfg.get("slice_dir"))
    parser.add_argument("--image_path", type=str, default=None, help="Force a single image (blank test)")
    parser.add_argument("--deployment", type=str, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--image_filename", type=str, default="Axial.png")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of parallel API requests")
    parser.add_argument("--limit", type=int, default=None)

    main(parser.parse_args())