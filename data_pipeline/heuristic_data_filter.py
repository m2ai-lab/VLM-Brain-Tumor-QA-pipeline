import os
import pandas as pd 
import re

#Add dataframes to list
frames = []

frames = pd.concat(frames)
indecies = []

for idx, entry in frames.iterrows():
        skip = False #Variable used to determine if we need to skip the QA pair

        #exclude QA pairs with any of these words
        for keyword in ["postsurgical change", "Postsurgical change", "postsurgical changes", "Postsurgical changes", "postsurgical", "Postsurgical", "retrospect", "Retrospect", "progression", 
"Progression", "recurrent", "Recurrent", "stable", "Stable", "tumor growth", "Tumor growth", "tumor shrinkage", "retrospective", "Retrospective",
"Tumor shrinkage", "metastasis", "Metastasis", "spine", "Spine", "spinal", "Spinal", "\\*\\*\\*\\*\\*", "\\*\\*\\*\\*\\*'s", "metastases", "Metastases",
"diffus\\*", "perfus\\*", "flow", "MRA", "MRV", "metasta\\*", "contrast", "age", "date", "fiducial", "discussed", "specified", "purpose", "radiation", "susceptibility",
"Diffus\\*", "Perfus\\*", "Flow", "Metasta\\*", "Contrast", "Age", "Date", "Fiducial", "Discussed", "Specified", "Purpose", "Radiation", "Susceptibility"]:
            
            if re.search(rf"{keyword}", entry['Question']) != None or re.search(rf"{keyword}", str(entry['Answer'])) != None:
                indecies.append(idx)

frames = frames.drop(indecies)
print(frames.shape)
frames.to_csv('path/to/ucsf_neuroimaging_pairs_0_9000.csv')