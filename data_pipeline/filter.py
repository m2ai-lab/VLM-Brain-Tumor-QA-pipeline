import pandas as pd 
import re

pairs = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/ucsf_pdgm_pairs.csv")
print(pairs.shape)  

keywords = ["postsurgical change", "Postsurgical change", "postsurgical changes", "Postsurgical changes", "postsurgical", "Postsurgical", "retrospect", "Retrospect", "progression", 
"Progression", "recurrent", "Recurrent", "stable", "Stable", "tumor growth", "Tumor growth", "tumor shrinkage", "retrospective", "Retrospective",
"Tumor shrinkage", "metastasis", "Metastasis", "spine", "Spine", "spinal", "Spinal", "\\*\\*\\*\\*\\*", "\\*\\*\\*\\*\\*'s", "metastases", "Metastases",
"diffus\\*", "perfus\\*", "flow", "MRA", "MRV", "metasta\\*", "contrast", "age", "date", "fiducial", "discussed", "specified", "purpose", "radiation", "susceptibility",
"Diffus\\*", "Perfus\\*", "Flow", "Metasta\\*", "Contrast", "Age", "Date", "Fiducial", "Discussed", "Specified", "Purpose", "Radiation", "Susceptibility"]

idx = 0
while idx < pairs.shape[0]:
    currow = pairs.iloc[idx]

    for keyword in keywords:
            if re.search(rf"{keyword}", str(currow['Question'])) != None or re.search(rf"{keyword}", str(currow['Answer'])) != None:
                    pairs = pairs.drop(pairs.index[idx])
                    continue

    idx += 1

print(pairs.shape)   
pairs.to_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")