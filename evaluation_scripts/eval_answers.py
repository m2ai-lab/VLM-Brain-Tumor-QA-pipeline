import argparse
import pandas as pd
import os
from collections import defaultdict
import json

def argument_handler():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--qa_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs")
    parser.add_argument('--answer_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/Evals.json")
    
    return parser.parse_args()

def main(args):
    answer_df = pd.read_csv(args.answer_path)
    accuracy = {}
    results_files = defaultdict(list)
    
    # 1. Build the dictionary of files
    for root, _, files in os.walk(args.qa_path):
        for file in files:
            if file.lower().endswith(".csv") and "result" in file.lower():
                test_name = root.split('/')[-1] +"_" + file.lower().split('.')[0].replace("_results","")
                read_path = os.path.join(root, file)
                write_path = os.path.join(root, test_name + "_wrongs.csv")
                results_files[test_name].append((read_path, write_path))

    # 2. Evaluate the results
    for test_name, file_info_list in results_files.items():
        # file_info_list is a list of tuples. We grab the first tuple:
        read_path, write_path = file_info_list[0] 
        
        results_df = pd.read_csv(read_path)
        print(f'Evaluating {test_name} for file {read_path}')
        
        rights = []
        wrongs = []
        indexes = []
        total_right = 0  # Initialize total_right to 0 for each test!
        
        idx = 0
        for ans, pred in zip(answer_df['Answer'], results_df['predicted_answer']):
            answer = str(ans)
            prediction = str(pred)
            
            if answer in prediction:
                total_right += 1
            else:
                indexes.append(idx)
                rights.append(answer)
                wrongs.append(prediction)
            
            idx+=1

        # 3. Save the wrongs cleanly using Pandas instead of numpy
        wrongs_df = pd.DataFrame({
            'Question Index': indexes,
            'Correct_Answer': rights, 
            'Predicted_Answer': wrongs
        })
        wrongs_df.to_csv(write_path, index=False)

        # 4. Calculate and store accuracy
        acc = total_right / len(results_df)
        accuracy[test_name] = acc
        print(f'{test_name} got accuracy: {acc}')
    
    # 5. Save final accuracy JSON
    json_string = json.dumps(accuracy, indent=4)
    with open(args.output_path, "w") as f:
        f.write(json_string)


if __name__ == "__main__":
    main(argument_handler())