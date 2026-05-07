import os
import kagglehub
import sys
from config_utils import load_config

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def download_vqa(target_dir):
    """
    Downloads VQA dataset directly from Kaggle Hub
    """
    print(f"--- Automated VQA Download to {target_dir} ---")
    
    # Download latest version
    path = kagglehub.dataset_download("redactedname29/redacted-vqa",output_dir=target_dir)

    print("Path to dataset files:", path)

    print("Download complete.")

if __name__ == "__main__":
    target_dir = os.path.dirname(_cfg.get("qa_path"))
    download_vqa(target_dir)
