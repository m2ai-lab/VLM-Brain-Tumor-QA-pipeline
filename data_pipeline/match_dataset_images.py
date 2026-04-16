import argparse
import pandas as pd
import os

def argument_parser():
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', 
                        type=str, 
                        required=False, 
                        help='Path to QA data',
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    parser.add_argument('--img_dir', 
                        type=str, 
                        required=False, 
                        help='Path to image directory',
                        default="/mnt/scratch/user/shghosh/UCSF-PDGM-v5")
    return parser.parse_args()

def main(args):

    qa_data = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv")
    ids = qa_data['Assigned ID'].unique()

    tree = os.listdir("/mnt/scratch/user/shghosh/UCSF-PDGM-v5")
    tree = [x.replace("_nifti", "") for x in tree]

    new_df = qa_data.copy()

    missing_ids = set(ids) - set(tree)
    print(f"Missing IDs: {missing_ids}")
    new_df = new_df[~new_df['Assigned ID'].isin(missing_ids)]

    new_df.to_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs_matched.csv", index=False)
    print(f"Saved matched data to /scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs_matched.csv")


if __name__ == "__main__":

    
    main(argument_parser())