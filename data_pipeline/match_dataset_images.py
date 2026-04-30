import argparse
import pandas as pd
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()



def argument_parser():
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', 
                        type=str, 
                        required=False, 
                        help='Path to QA data',
                        default=_cfg.get("qa_path"))
    parser.add_argument('--img_dir', 
                        type=str, 
                        required=False, 
                        help='Path to image directory',
                        default=_cfg.get("nifti_root"))
    return parser.parse_args()

def main(args):

    qa_data = pd.read_csv(args.qa_path)
    ids = qa_data['Assigned ID'].unique()

    tree = os.listdir(args.img_dir)
    tree = [x.replace("_nifti", "") for x in tree]

    new_df = qa_data.copy()

    missing_ids = set(ids) - set(tree)
    print(f"Missing IDs: {missing_ids}")
    new_df = new_df[~new_df['Assigned ID'].isin(missing_ids)]

    path_without_filename = os.path.dirname(args.qa_path)
    output_filename = os.path.basename(args.qa_path).replace(".csv", "") + "_matched.csv"
    output_path = os.path.join(path_without_filename, output_filename)
    new_df.to_csv(output_path, index=False) 
    print(f"Saved matched data to {output_path}")


if __name__ == "__main__":

    
    main(argument_parser())