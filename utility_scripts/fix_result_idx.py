import pandas as pd



dataset = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")


for i in ["/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/match_question_format.py"]:
    rewrite_df = pd.read_csv(i)
    rewrite_df = rewrite_df.set_index('question_id')
    rewrite_df.index = dataset.iloc[:, 0]
    rewrite_df["Question"] = dataset["Question"]

    rewrite_df.index.name = 'question_id'
    rewrite_df.to_csv(i)

