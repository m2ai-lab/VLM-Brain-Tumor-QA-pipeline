import argparse
import pandas as pd
import os
import re
import random
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

def argument_parser():
    parser = argparse.ArgumentParser(description="Question Reshuffeling Script")
    parser.add_argument('--qa_path', type=str, required=False,
                        help='Path to QA data', default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to output data', default=_cfg.get("reshuffled_qa_path"))
    return parser.parse_args()

def reshuffle_question(mc_question):
    split_question = re.split(r"[0-5]\)",mc_question)
    question = split_question[0]
    options = split_question[1:]
    random.shuffle(options)
    options = [str(idx) + ")" + option for option,idx in zip(options,range(1,len(options)+1))]
    reshuffled_question = question + " ".join(options)
    return reshuffled_question.replace("   "," ")




def main(args):

    #Load in the QA data
    qa_data = pd.read_csv(args.qa_path)

    for idx,row in qa_data.iterrows():
        qa_data.loc[idx,"Question"] = reshuffle_question(row["Question"])

    qa_data.to_csv(args.output_path)



if __name__ == "__main__":
    main(argument_parser())