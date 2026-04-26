import pandas as pd
import json
import argparse
import tempfile
import subprocess
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def main(args):
    print(f"Loading QA CSV from: {args.qa_path}")
    df = pd.read_csv(args.qa_path)
    
    # 1. Generate LLaVA-Med JSONL input
    print("Converting QA rows into JSONL format expected by LLaVA-Med...")
    records = []
    
    for idx, row in df.iterrows():
        # Handle different variants/images (default single_slice is ID/Axial.png)
        # For 'human' or similar, we use the Assigned ID
        accession = row["Assigned ID"]
        
        # Determine image_name based on user's csv_to_llava_jsonl behavior
        # In single_slice it's commonly f"{accession}/Axial.png"
        image_name = f"{accession}/Axial.png"
        
        # If it's a blank run, image_name is the path directly
        if args.image_path:
            # Not natively supported by model_vqa --image-folder easily if it's just one file,
            # but we can pass the absolute path if needed, or we just rely on standard.
            pass
            
        record = {
            "question_id": idx,  # Use actual dataframe index so we can map back cleanly
            "image": image_name,
            "text": row["Question"].strip() + "\n<image>",
        }
        records.append(record)
        
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f_q:
        for r in records:
            f_q.write(json.dumps(r) + "\n")
        temp_q = f_q.name
        
    print(f"Temp questions JSONL created at {temp_q}")
    
    # temp file for answers
    temp_a = temp_q.replace(".jsonl", "_answers.jsonl")
    
    # 2. Call LLaVA-Med inference script
    print("Executing LLaVA-Med VQA evaluation...")
    cmd = [
        "python", "-m", "llava.eval.model_vqa",
        "--conv-mode", "mistral_instruct",
        "--model-path", args.model_path,
        "--question-file", temp_q,
        "--image-folder", args.image_dir,
        "--answers-file", temp_a
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run from the LLaVA-Med repo so that `-m llava` works correctly
    try:
        subprocess.run(cmd, cwd=args.llava_repo_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"LLaVA-Med inference failed with exit code {e.returncode}")
        # Clean up and exit
        os.remove(temp_q)
        if os.path.exists(temp_a):
            os.remove(temp_a)
        exit(1)
        
    # 3. Parse JSONL outputs back into CSV
    #    LLaVA-Med model_vqa writes JSONL in some versions, CSV in others.
    #    Auto-detect by inspecting the first line.
    print(f"Parsing answers from {temp_a} and converting back to CSV format...")
    responses = {}

    with open(temp_a, "r") as f:
        first_line = f.readline().strip()

    if first_line.startswith("question_id,"):
        # ── CSV format (some LLaVA-Med forks write this) ────────────────────
        import csv as _csv
        with open(temp_a, "r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    responses[int(row["question_id"])] = row.get("predicted_answer", "Error")
                except (ValueError, KeyError):
                    pass
        print(f"  Detected CSV format — parsed {len(responses)} answers.")
    else:
        # ── JSONL format ─────────────────────────────────────────────────────
        bad_lines = 0
        with open(temp_a, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    responses[data["question_id"]] = data.get("text", "Error")
                except json.JSONDecodeError:
                    bad_lines += 1
                    print(f"  [warn] Skipped non-JSON line {i}: {line[:80]!r}")
        if bad_lines:
            print(f"  [warn] Skipped {bad_lines} non-JSON lines in answer file.")
        print(f"  Detected JSONL format — parsed {len(responses)} answers.")

            
    generated_answer = []
    for idx in df.index:
        generated_answer.append(responses.get(idx, "No Answer Generated"))
        
    df["predicted_answer"] = generated_answer
    df["LLaVA_Reasoning"] = "" # LLaVA outputs raw text, no structured reasoning split
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    df.to_csv(args.output_path, index=False)
    print(f"Successfully saved LLaVA-Med output to {args.output_path}!")
    
    # Clean up
    os.remove(temp_q)
    # os.remove(temp_a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-Med Inference Wrapper")
    parser.add_argument('--qa_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--image_path', type=str, default=None, help="Not used for full test, here for blank compatibility")
    parser.add_argument('--model_path', type=str, default=_cfg.get("llavamed_model_path"))
    parser.add_argument('--llava_repo_dir', type=str, default=_cfg.get("llavamed_repo_dir"))

    args = parser.parse_args()
    main(args)
