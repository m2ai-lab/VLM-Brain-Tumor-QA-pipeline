import argparse
import pandas as pd
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config_utils import load_config
_cfg = load_config()

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

    # ── DEBUG: id_look_up vs qa_data['DeID Note Key'] ───────────────────
    print('\n[DEBUG] id_look_up size:', len(id_look_up))
    sample_keys = list(id_look_up.keys())[:5]
    print('[DEBUG] Sample id_look_up keys (deid_accession_number):', sample_keys)
    print('[DEBUG] Sample qa_data DeID Note Key values:',
          qa_data['DeID Note Key'].head(5).tolist())
    print(f'[DEBUG] qa_data DeID Note Key dtype: {qa_data["DeID Note Key"].dtype} | '
          f'id_look_up key type: {type(sample_keys[0]) if sample_keys else "N/A"}')

    # Map DeID Note Key to Identified_Accession_Number
    qa_data['Identified_Accession_Number'] = qa_data['DeID Note Key'].map(id_look_up)

    nan_count = qa_data['Identified_Accession_Number'].isna().sum()
    print(f'\n[DEBUG] After mapping DeID Note Key -> Identified_Accession_Number: '
          f'{nan_count}/{len(qa_data)} rows are NaN (no match in id_look_up)')
    mapped_sample = qa_data['Identified_Accession_Number'].dropna().head(5).tolist()
    print('[DEBUG] Sample Identified_Accession_Number values after map:', mapped_sample)

    if args.preferred_only:
        seq_type_data = seq_type_data[seq_type_data['preferred'] == 'True']

    if args.filter_types:
        seq_type_data = seq_type_data[seq_type_data['sequence'].isin(args.filter_types)]

    seq_type_data['Identified_Accession_Number'] = seq_type_data['seq_dir'].astype(str).str.split('/').str[5]

    # ── DEBUG: seq_dir parsing ───────────────────────────────────────────
    print('\n[DEBUG] Sample seq_dir values (first 3):')
    for p in seq_type_data['seq_dir'].head(3).tolist():
        parts = str(p).split('/')
        print(f'  path: {p}')
        print(f'  parts: {parts}')
        print(f'  index 5 gives: "{parts[5] if len(parts) > 5 else "OUT OF RANGE (only " + str(len(parts)) + " parts)"}"')
    print('[DEBUG] Sample Identified_Accession_Number extracted (index 5):',
          seq_type_data['Identified_Accession_Number'].head(5).tolist())
    print(f'[DEBUG] Unique Identified_Accession_Number count in seq_type_data: '
          f'{seq_type_data["Identified_Accession_Number"].nunique()}')

    # 1. Output the distinct questions
    # Get accessions that actually have sequences
    valid_accessions = seq_type_data['Identified_Accession_Number'].unique()
    
    # ── DEBUG: overlap between qa_data and seq_type_data ────────────────
    qa_ids  = set(qa_data['Identified_Accession_Number'].dropna().unique())
    seq_ids = set(valid_accessions)
    overlap = qa_ids & seq_ids
    print(f'\n[DEBUG] QA unique Identified_Accession_Number (non-NaN): {len(qa_ids)}')
    print(f'[DEBUG] seq_type_data unique Identified_Accession_Number:  {len(seq_ids)}')
    print(f'[DEBUG] Overlap (accessions in BOTH):                       {len(overlap)}')
    if overlap:
        print('[DEBUG] Sample overlapping IDs:', list(overlap)[:5])
    else:
        print('[DEBUG] NO OVERLAP -- this is why found_qs is empty.')
        print('[DEBUG] Sample QA IDs:  ', list(qa_ids)[:5])
        print('[DEBUG] Sample seq IDs:', list(seq_ids)[:5])

    # Filter questions that exist in valid_accessions
    found_qs = qa_data[qa_data['Identified_Accession_Number'].isin(valid_accessions)].copy()
    final_data = found_qs[["Question", "Answer", "Assigned ID", "Identified_Accession_Number"]]

    print(f'\nQA Output shape (Unique Questions) {final_data.shape[0]} x {final_data.shape[1]}')
    final_data.to_csv(args.output_path, index=False)

    # 2. Output the mapping list
    if args.single_dicom:
        def get_first_dcm(path):
            if pd.isna(path) or str(path) == 'nan' or not os.path.exists(str(path)):
                return None
            for file in os.listdir(path):
                if file.endswith(".dcm"):
                    return os.path.join(path, file)
            return None
        seq_type_data['image_path'] = seq_type_data['seq_dir'].apply(get_first_dcm)
    else:
        seq_type_data['image_path'] = seq_type_data['seq_dir']

    seq_type_data['image_path'] = seq_type_data['image_path'].str.replace('neuroimaging_data', 'neuroimaging_refresh', regex=False)
    seq_type_data = seq_type_data.dropna(subset=['image_path'])

    # Only map ones relevant to QA data
    scan_mapping = seq_type_data[seq_type_data['Identified_Accession_Number'].isin(found_qs['Identified_Accession_Number'])].copy()
    
    rev_id_look_up = {v: k for k, v in id_look_up.items()}
    scan_mapping['Deidentified_Accession_Number'] = scan_mapping['Identified_Accession_Number'].map(rev_id_look_up)
    
    scan_mapping = scan_mapping[['Deidentified_Accession_Number', 'sequence', 'image_path', 'Identified_Accession_Number']]
    
    print(f'Scan Mapping Output shape {scan_mapping.shape[0]} x {scan_mapping.shape[1]}')
    scan_mapping.to_csv(args.scan_mapping_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QMRI Match Script")
    parser.add_argument('--qa_path', type=str, required=False, help='Path to QA data',
                        default=_cfg.get("qa_path"))
    parser.add_argument('--output_path', type=str, required=False, help='Path to output data',
                        default=_cfg.get("output_base", "") + "/dicom_dataset.csv")
    parser.add_argument('--scan_mapping_path', type=str, required=False,
                        help='Path to output scan mapping data',
                        default=_cfg.get("output_base", "") + "/scan_mapping.csv")
    parser.add_argument('--deid_to_newID_path', type=str, required=False,
                        help='Path to de-identified to new imaging ID mapping',
                        default=_cfg.get("deid_to_newID_path"))
    parser.add_argument('--newID_to_accession_path', type=str, required=False,
                        help='Path to new imaging ID to accession code mapping',
                        default=_cfg.get("newID_to_accession_path"))
    parser.add_argument('--seq_type_path', type=str, required=False,
                        help='Path to where sequence type data is stored',
                        default=_cfg.get("seq_type_path"))
    parser.add_argument('--filter_types', nargs='*', required=False,
                        help='Type of scan you would like to get a subset of from data. Leave empty for all sequence types.',
                        default=[])
    parser.add_argument('--preferred_only', type=bool, required=False,
                        help='If True, only retains sequences explicitly marked as preferred',
                        default=False)
    parser.add_argument('--num_entries', type=int, required=False,
                        help='Limit the size of the dataset if required',
                        default=100000)
    parser.add_argument('--single_dicom', type=bool, required=False,
                        help='single_dicom if required',
                        default=False)
    args = parser.parse_args()

    main(args)