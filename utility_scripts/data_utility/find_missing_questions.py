import argparse
import pandas as pd
import re
import os

def argument_handler():
    parser = argparse.ArgumentParser(description="Creating blank Nifti Pipeline")
    parser.add_argument('--response_path', type=str, required=False, default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    parser.add_argument('--ground_truth_path', type=str, required=False, default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    return parser.parse_args()

def main(args):
    response_df = pd.read_csv(args.response_path)
    ground_truth_df = pd.read_csv(args.ground_truth_path)

    output_df = pd.DataFrame()
    response_df["Question"] = [None]*len(response_df)
    ground_truth_df["predicted_answer"] = ["Invalid Response"] * len(ground_truth_df)
    
    #Get all correct Answers for a question from the response df

    output_rows = []
    for _,row in response_df.iterrows():
        assigned_id = row['question_id']
        answer = row['predicted_answer']
        possible_questions = ground_truth_df[ground_truth_df["Assigned ID"] == assigned_id]["Question"]
        for possible_question in possible_questions:
            if answer in possible_question:
                response_df.loc[response_df['question_id'] == assigned_id, 'Question'] = possible_question
                break
    for _,row in ground_truth_df.iterrows():
        if row["Question"] not in response_df["Question"].values:
            output_rows.append(row)
        else:
            output_rows.append(response_df.loc[response_df['Question'] == row["Question"]])

    output_df = pd.DataFrame(output_rows)
    output_dir = os.path.dirname(args.response_path)
    output_filename = os.path.basename(args.response_path)
    output_filename = output_filename.replace(".csv", "_matched.csv")
    output_df.to_csv(os.path.join(output_dir, output_filename), index=False)


if __name__ == "__main__":
    main(argument_handler())