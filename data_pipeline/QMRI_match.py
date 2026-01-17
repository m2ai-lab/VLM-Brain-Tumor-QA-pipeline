import argparse

def main(args):
    import pandas as pd

    #Load in the QA data
    qa_data = pd.read_excel(args.qa_path)
    print('Loaded QA data with shape:', qa_data.shape)

    deid_to_newID = pd.read_csv(args.deid_to_newID_path, 
                                usecols=['newID', 'deid_accession_number'],
                                dtype=str)

    print('Loaded deid to newID mapping with shape:', deid_to_newID.shape)

    newID_to_accession = pd.read_csv(args.newID_to_accession_path, 
                                     usecols=['newID', 'accession_number'],
                                     dtype=str)

    print('Loaded newID to accession mapping with shape:', newID_to_accession.shape)

    deid_to_accession = pd.merge(deid_to_newID, 
                                 newID_to_accession, 
                                 left_on='newID',
                                 right_on='newID',
                                 how='left', 
                                 validate='1:1').set_index("deid_accession_number").drop('newID', axis=1)

    print('Merged deid to accession data, resulting shape:', deid_to_accession.shape)


    notes_to_acc = pd.merge(qa_data, 
                            deid_to_accession, 
                            left_on='Note ID', 
                            right_on='deid_accession_number', 
                            how='left',
                            validate='many_to_one').set_index('accession_number')
    print('Merged notes with accession data, resulting shape:', notes_to_acc.shape)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', 
                        type=str, 
                        required=False, 
                        help='Path to QA data',
                        default="/home/remote/shghosh/QApairs_NewPrompt_temp.xlsx")
    parser.add_argument('--output_path', 
                        type=str, 
                        required=False, 
                        help='Path to output data',
                        default="~")
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