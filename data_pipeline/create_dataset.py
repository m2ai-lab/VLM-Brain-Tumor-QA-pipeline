import argparse
import pandas as pd
import os

def load_datasets(args):
    qa_data = pd.read_csv(args.qa_path, index_col=0)

    seq_type_data = pd.read_csv(args.seq_type_path,
                                usecols=['preferred', 'seq_dir', 'sequence'],
                                dtype=str)
    deid_to_newID = pd.read_csv(args.deid_to_newID_path, 
                                usecols=['newID', 'deid_accession_number'],
                                dtype=str)
    newID_to_accession = pd.read_csv(args.newID_to_accession_path, 
                                    usecols=['newID', 'accession_number'],
                                    dtype=str)

    return qa_data,seq_type_data,deid_to_newID,newID_to_accession
def main(args):

    #Load in the QA data
    qa_data,seq_type_data,deid_to_newID,newID_to_accession = load_datasets(args)

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

    seq_type_data = seq_type_data[(seq_type_data['preferred'] == 'True') & (seq_type_data['sequence'].isin(args.filter_types))]

    seq_type_data['Accession_number'] = seq_type_data['seq_dir'].astype(str).str.split('/').str[5]

    accession_to_path = dict(zip(seq_type_data['Accession_number'], seq_type_data['seq_dir']))

    # 2. Map the paths to a temporary variable
    folder_paths = qa_data['Accession_number'].map(accession_to_path)

    if args.single_dicom:
        def get_first_dcm(path):
            # Check if the path is valid and exists
            if pd.isna(path) or not os.path.exists(path):
                return None
            
            # Return the full path of the first .dcm file found
            for file in os.listdir(path):
                if file.endswith(".dcm"):
                    return os.path.join(path, file)
                    
            return None
        

        # 4. Apply it to create your new column
        qa_data['image_path'] = folder_paths.apply(get_first_dcm)
    
    else:
        qa_data['image_path'] = folder_paths

    #Replaced 'accession_num' with 'Accession Num'
    qa_data = qa_data.rename(columns={'Accession Num': 'Deidentified_Accession_Number'})

    final_data = qa_data[["Question", "Answer", "image_path","Deidentified_Accession_Number"]]

    print("Selected only QA paris with FLAIR scans found")
    found_flair = final_data.dropna(subset=['image_path'])

    print("Get subset of total for formatting dataset")
    # found_flair = found_flair.head(args.num_entries)

    print(f'Output shape {found_flair.shape[0]} x {found_flair.shape[1]}')
    found_flair.to_csv(args.output_path, index=False)


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
                        default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/dicom_dataset.csv")
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
    parser.add_argument('--seq_type_path', 
                        type=str, 
                        required=False, 
                        help='Path to where sequence type data is stored',
                        default="/mnt/fac/CX000019_DS1/neuroimaging_seqtype.csv")
    parser.add_argument('--filter_types',
                        nargs='+',
                        required=False, 
                        help='Type of scan you would like to get a subset of from data',
                        default=["FLAIR"])

    parser.add_argument('--num_entries',
                        type=int, 
                        required=False, 
                        help='Limit the size of the dataset if required',
                        default=100000)
    parser.add_argument('--single_dicom',
                        type=bool, 
                        required=False, 
                        help='single_dicom if required',
                        default=False)
    args = parser.parse_args()
    
    main(args)