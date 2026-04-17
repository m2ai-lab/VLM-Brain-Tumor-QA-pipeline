import os
import pandas as pd
import argparse

def argument_handler():
    parser = argparse.ArgumentParser(description="Filter out specific rows from result CSVs in a directory.")
    parser.add_argument(
        '--dir_path', 
        type=str, 
        required=True, 
        help="Path to the directory you want to walk through."
    )
    parser.add_argument(
        '--target_id', 
        type=str, 
        required=False, 
        default="UCSF-PDGM-0348",
        help="The string/ID to search for and remove."
    )
    return parser.parse_args()

def main(args):
    target = args.target_id
    directory = args.dir_path
    
    # 1. Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            # 2. Look for CSVs with "_results" in the name
            if file.endswith(".csv") and "_results" in file:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    # 3. Load the CSV into a DataFrame
                    df = pd.read_csv(file_path)
                    original_len = len(df)
                    
                    # 4. Filter out any row containing the target string
                    # We cast to string to ensure we don't get TypeErrors on numeric columns, 
                    # then apply a string check across all columns, checking row-wise (axis=1).
                    mask = df.astype(str).apply(lambda col: col.str.contains(target, case=False, na=False)).any(axis=1)
                    
                    # Keep only the rows where the mask is False (meaning the target was NOT found)
                    filtered_df = df[~mask]
                    
                    dropped_count = original_len - len(filtered_df)
                    
                    # 5. Save if modifications were made
                    if dropped_count > 0:
                        print(f"  -> Dropped {dropped_count} rows containing '{target}'.")
                        
                        filtered_df.to_csv(file_path, index=False)
                        print(f"  -> Saved filtered file to: {file_path}\n")
                    else:
                        print("  -> No target rows found. Skipping.\n")
                        
                except Exception as e:
                    print(f"  -> Error processing {file_path}: {e}\n")

if __name__ == "__main__":
    main(argument_handler())