import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer   
import argparse
from pydantic import BaseModel, Field, ValidationError
import re

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint, get_row_id
_cfg = load_config()

# Added a strong system prompt to enforce JSON output
SYSTEM_PROMPT = """
You are an expert radiologist AI. You must output your response ONLY as a valid JSON object.
Do not include any conversational text before or after the JSON. 
Use the following format:
{
  "reasoning": "Your step-by-step clinical reasoning here.",
  "answer": "The final choice selected from the options."
}
"""

FEW_SHOT_EXAMPLE = """
Example Request:
Question: Based on the T2/FLAIR hyperintensity, what is the most likely grade? 1) Low Grade 2) High Grade

Example Response:
{
  "reasoning": "The slices show significant mass effect and central necrosis within the T1-contrast enhancing lesion, which is highly suggestive of aggressive growth.",
  "answer": "2) High Grade"
}
"""

class QwenResponse(BaseModel):
    reasoning: str = Field(description="Reasoning for answer")
    answer: str = Field(description="The final choice selected from the options.")

def clean_json_string(raw_str):
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str


def _parse_response(output_text: str) -> dict:
    cleaned_response = clean_json_string(output_text)
    try:
        if not cleaned_response:
            raise ValueError("Regex failed to find anything resembling JSON.")
        validated_data = QwenResponse.model_validate_json(cleaned_response)
        return validated_data.model_dump()
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": output_text}
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error", "raw": output_text}


def run_batch(model, tokenizer, batch_rows: list[dict]) -> list[dict]:
    """
    Run batched text-only inference on a list of QA rows.

    Qwen has no image input — each row's question is independently tokenized
    and padded into a single batch tensor.  Left-padding ensures generation
    is not confused by trailing padding tokens.
    """
    texts: list[str] = []
    for row in batch_rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{FEW_SHOT_EXAMPLE}\n====\nQuestion: {row['Question']}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    # Left-padding required for batched generation with decoder-only models
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    input_length = inputs['input_ids'].shape[1]
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

    results: list[dict] = []
    for i in range(len(batch_rows)):
        new_tokens = generated_ids[i][input_length:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(_parse_response(output_text))

    return results


def main(args):
    print(f"Loading Qwen from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # bf16 is great for newer models like Qwen to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    qa_data = pd.read_csv(args.qa_path)

    # If shuffled is specified, use the "Shuffled Question" column
    if getattr(args, "shuffled", False):
        if "Shuffled Question" in qa_data.columns:
            print("Using 'Shuffled Question' column instead of 'Question' as requested.")
            qa_data["Question"] = qa_data["Shuffled Question"]
        else:
            print("WARNING: --shuffled specified but 'Shuffled Question' column not found. Using original questions.")

    completed_ids = load_checkpoint(args.output_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} rows already completed, skipping.")

    total = len(qa_data)
    batch_size = args.batch_size
    print(f"Running Qwen inference with batch_size={batch_size} on {total} rows.")

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
        print(f"Processing rows {batch_start + 1}–{batch_end}/{total} "
              f"(batch of {len(pending_records)})...")

        responses = run_batch(model, tokenizer, list(pending_records))

        batch_df = qa_data.iloc[list(pending_df_indices)].copy()
        save_checkpoint(
            args.output_path,
            batch_df,
            {
                "predicted_answer": [r.get("answer",    "Error") for r in responses],
                "Qwen_Reasoning":   [r.get("reasoning", "Error") for r in responses],
            },
        )
        completed_ids.update(
            str(rec.get("Assigned ID", i)) for i, rec in zip(pending_df_indices, pending_records)
        )
        print(f"  → Checkpoint saved ({len(completed_ids)}/{total} total done).")

    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Text Inference")
    parser.add_argument('--qa_path', type=str, default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, default=_cfg.get("output_base", "") + "/Qwen/text_only_results.csv")
    parser.add_argument('--model_path', type=str, default=_cfg.get("qwen_model_path"))
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Number of QA rows per model.generate() call. Text-only so higher is fine.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output and start fresh.")
    parser.add_argument('--shuffled', action='store_true', help="Use 'Shuffled Question' column if available.")

    args = parser.parse_args()
    main(args)