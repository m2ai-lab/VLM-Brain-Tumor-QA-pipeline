import argparse
import pandas as pd
import re
import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()


def argument_handler():
    parser = argparse.ArgumentParser(description="Match the origional dataset to the human dataset")
    parser.add_argument('--human_qa_path', type=str, required=False, default=_cfg.get("human_qa_path"))
    parser.add_argument('--input_results_path', type=str, required=False, default=None)
    return parser.parse_args()

def main(args):
    input_dataset = pd.read_csv(args.input_results_path)
    human_dataset = pd.read_csv(args.human_qa_path)

    # create a mapping of questions to their index in the dataset

    print("Serching if unique questions for all human dataset...")
    print(len(human_dataset["Question"].unique()) == len(human_dataset))

    print("Serching for if unique questions for input dataset...")
    print(len(input_dataset["Question"].unique()) == len(input_dataset))


    #Finding questions

    print("Creating mapping of questions to their index in the dataset...")
    qa_mapping = { row["Question"]:idx for idx,row in human_dataset.iterrows()}

    print("Finding questions in the input dataset that are present in the human dataset...")
    index = [qa_mapping[row["Question"]] if row["Question"] in qa_mapping.keys() else -1 for _,row in input_dataset.iterrows()]

    print("Assigning the index to the input dataset...")
    input_dataset["q_id"] = index

    print("Removing the questions that are not present in the human dataset...")
    input_dataset = input_dataset[input_dataset["q_id"] != -1]

    print("length of datasets aligned: ", len(human_dataset) == len(input_dataset))
    input_dataset.drop(columns=["q_id"], inplace=True)

    print("Saving the matched dataset...")
    new_human_path = args.input_results_path.replace("_results", "_human_results")
    input_dataset.to_csv(new_human_path, index=False)

    print("Done!")

if __name__ == "__main__":
    main(argument_handler())