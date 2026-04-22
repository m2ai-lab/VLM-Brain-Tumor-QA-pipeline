import argparse
import pandas as pd
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def main(args):
    qa_data = pd.read_csv(args.qa_path, index_col=0)
    origional_qa_data = pd.read_csv(args.origional_qa_path, index_col=0)

    output = pd.merge(qa_data, origional_qa_data, on=["Question", "Answer"], how="left")

    print("Origional columns: ", origional_qa_data.columns, "\nOrigional shape: ", origional_qa_data.shape)
    print("QA columns: ", qa_data.columns, "\nQA shape: ", qa_data.shape)
    print("Output columns: ", output.columns, "\nOutput shape: ", output.shape)
    columns = list(qa_data.columns) + ["Assigned ID"] 
    output = output[columns]

    output.to_csv(args.qa_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', type=str, required=False, help='Path to QA data',
                        default=_cfg.get("human_qa_path"))
    parser.add_argument('--origional_qa_path', type=str, required=False, help='Path to origional QA data',
                        default=_cfg.get("qa_path"))
    args = parser.parse_args()

    main(args)