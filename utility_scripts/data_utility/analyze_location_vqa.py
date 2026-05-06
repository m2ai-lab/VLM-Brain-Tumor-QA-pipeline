import pandas as pd
import glob
import os
import re
import sys
import os

# ── Dynamic Config Resolution ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:   
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def normalize(text):
    """
    Normalizes answer strings to handle cases like '2) Answer' vs 'Answer'.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove common prefixes like "1) ", "a. ", "1. "
    text = re.sub(r'^[a-z0-9][\s\).\-]+', '', text)
    return text.strip()

def run_analysis():
    results = {}
    
    base_path = _cfg.get("output_base")
    
    # Find all CSV files recursively
    # Filtering for 'text_only' in name and NOT 'wrongs'
    all_files = glob.glob(f"{base_path}/**/*text_only|blank*.csv", recursive=True)
    target_files = [f for f in all_files if "wrongs" not in f]
    
    if not target_files:
        print("No matching files found. Ensure you are running this in the parent directory of the model folders.")
        return

    for file_path in target_files:
        # Determine model name from directory or filename
        model_name = os.path.dirname(file_path).split(os.sep)[-1]
        if not model_name:
            model_name = "Root"
            
        if model_name not in results:
            results[model_name] = {'correct': 0, 'total': 0}
            
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required = ['Question', 'Answer', 'predicted_answer']
            if not all(col in df.columns for col in required):
                continue
                
            # Filter rows where Question contains "location" or "where" (case-insensitive)
            mask = df['Question'].str.contains(r'location|where', case=False, na=False)
            subset = df[mask]
            
            # Update denominator
            results[model_name]['total'] += len(subset)
            
            # Calculate accuracy for this subset
            # Using normalization to handle "1) Choice" vs "Choice"
            correct_count = subset.apply(
                lambda row: normalize(row['Answer']) == normalize(row['predicted_answer']), 
                axis=1
            ).sum()
            
            results[model_name]['correct'] += correct_count
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n" + "="*40)
    print("Location/Where Question Accuracy")
    print("="*40)
    for model, counts in sorted(results.items()):
        if counts['total'] > 0:
            print(f"{model}: {counts['correct']}/{counts['total']}")
        else:
            print(f"{model}: 0/0 (No 'location'/'where' questions found)")

if __name__ == "__main__":
    run_analysis()
