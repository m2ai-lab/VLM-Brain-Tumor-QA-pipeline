import argparse
import pandas as pd
import os

def load_datasets(args):
    qa_data = pd.read_csv(args.qa_path, index_col=0)

    deid_to_newID = pd.read_csv(args.deid_to_newID_path, 
                                usecols=['newID', 'deid_accession_number'],
                                dtype=str)
    newID_to_accession = pd.read_csv(args.newID_to_accession_path, 
                                    usecols=['newID', 'accession_number'],
                                    dtype=str)

    return qa_data,deid_to_newID,newID_to_accession
def main(args):

    #Load in the QA data
    qa_data,deid_to_newID,newID_to_accession = load_datasets(args)

    #'Note ID' switched to DeID Note Key
    qa_data = qa_data.dropna(subset=['DeID Note Key'])
    qa_data['DeID Note Key'] = qa_data['DeID Note Key'].astype(str).str.strip()
    print('Loaded QA data with shape:', qa_data.shape)


    deid_to_newID['deid_accession_number'] = deid_to_newID['deid_accession_number'].astype(str).str.strip()

    print('Loaded deid to newID mapping with shape:', deid_to_newID.shape)

    newID_to_accession = newID_to_accession.dropna(subset=['accession_number'])
    newID_to_accession['accession_number'] = newID_to_accession['accession_number'].astype(str).str.strip()
    print('Loaded newID to accession mapping with shape:', newID_to_accession.shape)

    deid_to_accession = pd.merge(deid_to_newID, 
                                 newID_to_accession, 
                                 left_on='newID',
                                 right_on='newID',
                                 how='left', 
                                 validate='1:1').drop('newID', axis=1)
    
    deid_to_accession = deid_to_accession.loc[:, ~deid_to_accession.columns.str.contains('^Unnamed')]

    print('Merged deid to accession data, resulting shape:', deid_to_accession.shape)
    
    notes_to_acc = pd.merge(qa_data, 
                            deid_to_accession, 
                            left_on='DeID Note Key', 
                            right_on='deid_accession_number', 
                            how='left',
                            validate='many_to_one')
    print('Merged notes with accession data, resulting shape:', notes_to_acc.shape)
    
    id_look_up = dict(zip(deid_to_accession['deid_accession_number'], 
                    deid_to_accession['accession_number']))

    # Now map it

    #Replace 'accession_num' with 'Accession Num'
    qa_data['Accession_number'] = qa_data['Accession Num'].map(id_look_up)


    # Replaced 'accession_num' with 'Accession Num'
    qa_data = qa_data.rename(columns={'Accession Num': 'Deidentified_Accession_Number'})

    print(f'QA Output shape (Unique Questions) {qa_data.shape[0]} x {qa_data.shape[1]}')

    # Find how many of the origional Assigned IDs i mapped
    origional_ids = qa_data["Assigned ID"].unique()
    mapped_ids = qa_data["Assigned ID"].unique()
    unmapped_ids = set(origional_ids) - set(mapped_ids)
    print(f'Number of unmapped IDs: {len(unmapped_ids)}')
    print(f'Unmapped IDs: {unmapped_ids}')
    
    mapped_ids = qa_data[["Assigned ID","Accession_number"]]
    mapped_ids = mapped_ids.drop_duplicates(subset=["Assigned ID"])
    mapped_ids.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', 
                        type=str, 
                        required=False, 
                        help='Path to QA data',
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")
    parser.add_argument('--output_path', 
                        type=str, 
                        required=False, 
                        help='Path to output data',
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/pdgm_tokens.csv")
    parser.add_argument('--deid_to_newID_path', 
                        type=str, 
                        required=False, 
                        help='Path to de-identified to new imaging ID mapping',
                        default="/mnt/fac/CX000019_DS1/CTSI/Radiology/De-Identified/RITM0609393A_Patient_Cohort_De_id_20251006.csv")
    parser.add_argument('--newID_to_accession_path', 
                        type=str, 
                        required=False, 
                        help='Path to new imaging ID to accession code mapping',
                        default="/mnt/fac/CX000019_DS1/CTSI/Radiology/Identified/RITM0609393A_Patient_Cohort_id_20251006.csv")  
    args = parser.parse_args()
    
    main(args)