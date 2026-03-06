import os
import sys
import socket
from collections import defaultdict
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask_sql import Context
from dask_jobqueue import SGECluster
from dask.distributed import Client, LocalCluster
import re

def remove_tag(response):
    match = re.search(r'```python\n+(.+)```', response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

cluster = LocalCluster(n_workers=8, memory_limit='128gb')
client = Client(cluster)

rwd_output = './assets/data/'

def load_register_table(data_asset, table, **kwargs):
    return dd.read_parquet(f'/wynton/protected/project/ic/data/parquet/{data_asset}/{table}/', **kwargs)

client.dashboard_link

note_meta = load_register_table("DEID_CDW", "note_metadata")
note_text = load_register_table("DEID_CDW", "note_text")


accessionnumbers = ['']
braintumor_meta = {}

for num in accessionnumbers:
    test = note_meta[note_meta['accessionnumber'] == num]
    test = test.compute()
    
    braintumor_meta[num] = test

clinical_notes = defaultdict(list)

for accessionnum, entries in braintumor_meta.items():
    for _, entry in entries.iterrows():
        currow = dict(entry.items())
        notes = note_text[note_text['deid_note_key'] == currow['deid_note_key']]
        notes = notes.compute()
        
        clinical_notes[accessionnum].append(dict(notes)['note_text'].iloc[0])


from openai import AzureOpenAI

#creating the dictioanry that will hold the question answer pairs
QAPairs = defaultdict(dict)

#Create the client that will be used to 
client = AzureOpenAI(api_key="",
                    api_version="2025-04-01-preview",
                    azure_endpoint="https://unified-api.ucsf.edu/general")

for accession, texts in clinical_notes.items():
    
    #Comebine all of the text
    all_text = ""
    for idx, text in enumerate(texts):
        all_text += 'Clinical Note (' + str(idx + 1) + ') :' + text + '\n\n'

    #Chat creation for making question answer pairs
    complete_llm = client.chat.completions.create(
            model="gpt-4o-2024-11-20", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You will be given the clinical notes \
                of a patient and will create 20 question answer pairs about the brain tumor patient.\
                Please return this as a python list and only a python list."},
                {"role": "user", "content": all_text},
            ]
        )

    partial_llm = client.chat.completions.create(
            model="gpt-4o-2024-11-20", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You will be given the clinical notes \
                of a patient. Please answer the following questions listed below\
                and also create more question answer pairs from these clinical notes (20 questions in total). \
                Please return this as a python list and only a python list. \
                \
                QUESTIONS: \
                 - Is there a tumor present based on the reports? \
                 - Where is the location of the tumor in ? \
                 - What are the signal characteristics on T1, T2, and FLAIR? \
                 - Is there evidence of necrosis or hemorrhage inside the tumor? \
                 - Are new lesions or metastases present compared to prior exams? \
                 - Is this change consistent with true progression or pseudoprogression? \
                 - Is this change consistent with treatment effect (e.g., radiation necrosis)? \
                 - Is this patient eligible for surgery, biopsy, or stereotactic radiosurgery based on lesion size and location?"},
                {"role": "user", "content": all_text},
            ]
        )


    categorized = client.chat.completions.create(
            model="gpt-4o-2024-11-20", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You will be given the clinical notes \
                of a patient and will create 20 question answer pairs about the brain tumor patient.\
                Please also categorize the question answer pairs into categorized based on the question asked.\
                Please return this as a python dictionary and only a python dictionary.\
                 \
                 EX: {'Category A': [(question1, answer 1)], 'Category B': [(question1, answer 1)]}"},
                {"role": "user", "content": all_text},
            ]
        )

    #get rid of python tags from responses
    #complete_llm = remove_tag(complete_llm.choices[0].message.content)
    #partial_llm = remove_tag(partial_llm.choices[0].message.content)
    #categorized = remove_tag(categorized.choices[0].message.content)
    
    #Add all of the QA responses with their respective reports
    QAPairs[accession]['report'] = all_text
    QAPairs[accession]['complete_llm'] = complete_llm.choices[0].message.content
    QAPairs[accession]['partial_llm'] = partial_llm.choices[0].message.content
    QAPairs[accession]['categorized'] = categorized.choices[0].message.content

#NOTES
# DO NOT INCLUDE
# questions about comparisons, clinical history, technique
# FOCUS ON Impression and findings