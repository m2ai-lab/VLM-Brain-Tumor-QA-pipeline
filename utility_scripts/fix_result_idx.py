import pandas as pd



dataset = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")

for i in ["/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/predicted_blank_results.csv"]:
    rewrite_df = pd.read_csv(i)
    rewrite_df = rewrite_df.set_index('question_id')
    rewrite_df.index = dataset.iloc[:, 0]

    rewrite_df.index.name = 'question_id'
    rewrite_df.to_csv(i)

