import os
import pandas as pd
import argparse
import sys
import os

# ── Dynamic Config Resolution ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def argument_handler():
    parser = argparse.ArgumentParser(description="Filter out specific rows from result CSVs in a directory.")
    parser.add_argument(
        '--dir_path', 
        type=str, 
        required=True, 
        help="Path to the directory you want to walk through."
    )
    parser.add_argument(
        '--row_indexes_to_drop', 
        type=list, 
        required=False, 
        default=[610, 611, 2307, 2429, 3017, 3193, 3194, 3195],
        help="List of row indexes to drop from each file."
    )
    return parser.parse_args()

def main(args):
    directory = args.dir_path
    
    # 1. Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            # 2. Look for CSVs with "_results" in the name
            if file.endswith(".csv") and "_results" in file:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    # 3. Load the CSV into a DataFrame
                    df = pd.read_csv(file_path)
                    original_len = len(df)
                    
                    df = df.drop(df.index[args.row_indexes_to_drop])
                    
                    df.to_csv(file_path, index=False)
                    print(f"  -> Saved filtered file to: {file_path}\n")
                        
                except Exception as e:
                    print(f"  -> Error processing {file_path}: {e}\n")

if __name__ == "__main__":
    main(argument_handler())