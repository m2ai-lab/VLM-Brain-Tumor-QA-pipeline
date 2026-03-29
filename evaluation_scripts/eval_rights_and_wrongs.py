import argparse
import pandas as pd
import os
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt
import json

def argument_handler():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--qa_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs")
    parser.add_argument('--answer_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")
    parser.add_argument('--output_top_wrong_figure_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_wrong_figure.png")
    parser.add_argument('--output_top_wrong_data_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_wrong.csv")
    parser.add_argument('--output_top_right_figure_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_right_figure.png")
    parser.add_argument('--output_top_right_data_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_right.csv")
    parser.add_argument('--exclude', nargs='+', default=["qwen","blank"]) # Use lowercase for easier matching
    parser.add_argument('--include', nargs='+', default=["/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/LLaVA_montage_wrongs.csv",
                                                         "/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/Med3DVLM/Med3DVLM_wrongs.csv",
                                                         "/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/MedGemma1.5/MedGemma1.5_multi_slice_wrongs.csv"
                                                         ]) # Use lowercase for easier matching
    return parser.parse_args()


def plot_counter(counter,output_path,plot_name):
    top = dict(counter.most_common(20))
    
    if not top:
        print("No wrong answers found, or no CSV files were processed.")
        return

    # Convert keys to strings so Seaborn treats them as categorical labels, not continuous numbers
    keys = [str(k) for k in top.keys()] 
    values = list(top.values())

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=keys, y=values, hue=keys, palette="Reds_r", legend=False)
    
    plt.title(plot_name)
    plt.xlabel("Question Index")
    plt.ylabel("Number of Times")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)

def question_counts_export(counter,output_path,answer_df):
    counter = dict(counter)

    q_indexes = [str(k) for k in counter.keys()] 
    counts = list(counter.values())

    questions = []
    answers = []
    
    for q_idx in q_indexes:
        idx = int(q_idx)
        questions.append(answer_df.loc[idx,"Question"])
        answers.append(answer_df.loc[idx,"Answer"])

    full_metrics = pd.DataFrame({"Count" :counts,"Question" : questions,"Answer" : answers})

    full_metrics.to_csv(output_path)


def main(args):
    # Load the master answer sheet to get the total number of questions dynamically
    answer_df = pd.read_csv(args.answer_path)
    answer_df = answer_df.rename(columns={answer_df.columns[0]: 'Question_Idx'})   
    answer_df = answer_df.set_index('Question_Idx') 
    
    all_questions = set(answer_df.index)
    print(all_questions)

    results_files = defaultdict(list)
    exclude_list = [ex.lower() for ex in args.exclude]
    
    # 1. Build the dictionary of files
    for root, _, files in os.walk(args.qa_path):
        # Skip this directory entirely if it matches our exclude list
        if any(ex in root.lower() for ex in exclude_list):
            continue
            
        for file in files:
            read_path = os.path.join(root, file)
            if read_path in args.include:
                file_lower = file.lower()
                if file_lower.endswith(".csv") and "wrongs" in file_lower:
                    # Ensure the filename itself doesn't contain an excluded word
                    if not any(ex in file_lower for ex in exclude_list):
                        test_name = root.split('/')[-1] + "_" + file_lower.split('.')[0].replace("results", "")
                        results_files[test_name].append(read_path)

    wrongs_hist = Counter()
    rights_hist = Counter()
    
    # 2. Evaluate the results
    for test_name, file_info_list in results_files.items():
        # Iterate over the string paths
        for file in file_info_list:
            df = pd.read_csv(file)
            
            if "Question Index" in df.columns:
                wrongs = set(df["Question Index"])
                wrongs_hist.update(list(df["Question Index"]))

                rights = all_questions - wrongs
                rights_hist.update(list(rights))

    
   
    plot_counter(wrongs_hist,args.output_top_wrong_figure_path,"Top 20 Most Frequently Missed Questions Across Evaluated Models")
    plot_counter(rights_hist,args.output_top_right_figure_path,"Top 20 Most Frequently Correct Questions Across Evaluated Models") 

    question_counts_export(wrongs_hist,args.output_top_wrong_data_path,answer_df)
    question_counts_export(rights_hist,args.output_top_right_data_path,answer_df)
    

if __name__ == "__main__":
    main(argument_handler())