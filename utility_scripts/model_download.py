"""
model_download.py — Utility to download model weights from HuggingFace to the models/ directory.
Uses snapshot_download to fetch entire model repositories.
"""
import os
import argparse
from huggingface_hub import snapshot_download

# Define the models we use and their target subdirectories within models/
# Format: { "HuggingFace/Repo-ID": "local_subdir_name" }
MODELS_TO_DOWNLOAD = {
    "google/medgemma-1.1-4b-it": "medgemma-1.5-4b-it", # Mapping to config name
    "lingshu-medical-mllm/Lingshu-32B": "Lingshu-32B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "lion-ai/MedImageInsights": "MedImageInsights",
    
}

def download_models(models_base_dir, specific_model=None):
    """
    Downloads models from HuggingFace.
    """
    os.makedirs(models_base_dir, exist_ok=True)
    
    to_download = MODELS_TO_DOWNLOAD
    if specific_model:
        if specific_model in MODELS_TO_DOWNLOAD:
            to_download = {specific_model: MODELS_TO_DOWNLOAD[specific_model]}
        else:
            print(f"Model '{specific_model}' not in predefined list. Attempting direct download...")
            to_download = {specific_model: specific_model.split("/")[-1]}

    for repo_id, subdir in to_download.items():
        target_dir = os.path.join(models_base_dir, subdir)
        print(f"\n--- Downloading {repo_id} to {target_dir} ---")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False, # Better for cluster environments
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # Save space/time
            )
            print(f"Successfully downloaded {repo_id}")
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VQA models from HuggingFace")
    parser.add_argument("--models_dir", type=str, default=None, 
                        help="Target directory for models (defaults to project_root/models)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific HF repo ID to download (e.g. google/medgemma-1.1-4b-it)")
    
    args = parser.parse_args()

    # Determine models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        # Fallback to local models/ directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")

    download_models(models_dir, specific_model=args.model)