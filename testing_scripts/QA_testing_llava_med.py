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
        
        # Determine image filename based on variant flag
        image_name = f"{accession}/{args.image_filename}"
        
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

    # 2. Call LLaVA-Med inference script
    print("Executing LLaVA-Med VQA evaluation...")
    cmd = [
        "python", "-m", "llava.eval.model_vqa",
        "--conv-mode", "mistral_instruct",
        "--model-path", args.model_path,
        "--question-file", temp_q,
        "--image-folder", args.image_dir,
        "--answers-file", args.output_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run from the LLaVA-Med repo so that `-m llava` works correctly
    try:
        subprocess.run(cmd, cwd=args.llava_repo_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"LLaVA-Med inference failed with exit code {e.returncode}")
        # Clean up and exit
        os.remove(temp_q)
        exit(1)

    # Clean up
    os.remove(temp_q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-Med Inference Wrapper")
    parser.add_argument('--qa_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--image_filename', type=str, default="Axial.png",
                        help="Filename inside each patient dir. Use 'axial_slices_montage.png' for montage.")
    parser.add_argument('--image_path', type=str, default=None, help="Not used for full test, here for blank compatibility")
    parser.add_argument('--model_path', type=str, default=_cfg.get("llavamed_model_path"))
    parser.add_argument('--llava_repo_dir', type=str, default=_cfg.get("llavamed_repo_dir"))

    args = parser.parse_args()
    main(args)
