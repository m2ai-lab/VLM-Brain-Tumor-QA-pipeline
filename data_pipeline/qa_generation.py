import pandas as pd 
from openai import AzureOpenAI
import json
import re

#grab the notes that were saved and convert them into a list for easy row access
clinical_notes = pd.read_csv('path/to/neuroimaging_notes_nopdgm.csv',)
clinical_notes = clinical_notes.iloc[5000:7000]

#creating the dictioanry that will hold the question answer pairs
QAPairs = {'Accession Num': [], 'Patient Pic ID': [], 'Patient Durable Key': [], 'DeID Note ID': [], 'DeID Note CSN ID': [], 'Procedure ID': [], 'DeID Note Key': [], 'Original Note': [], 'Question': [], 'Answer': [], 'Tag': []}

#Create the client that will be used to create QA pairs
client = AzureOpenAI(api_key="",
                    api_version="2025-04-01-preview",
                    azure_endpoint="https://unified-api.ucsf.edu/general")

#Create questions for all notes
for idx, row in clinical_notes.iterrows():

    #Chat creation for making question answer pairs with complete help from the LLM and sorted in categories
    categorized_prompt = f"""
    You will be given the radiology report of a patient.
    Your job is to create question-answer pairs based on the information given in the 'IMPRESSION' and 'FINDING' sections of the report. 
    
    Each question MUST have 4 options, 1 correct option and 3 incorrrect options, and these options MUST be in the same string as the question. 
    You MUST create 20 question-answer pairs

    EXAMPLE QUESTION FORMAT:
    Where is the location of the tumor?
    1) Upper Left Region  2) Upper Right Region  3) Lower Left Region  4) Lower Right Region
    
    EXAMPLE ANSWER FORMAT:
    Upper Right Region
    
    CREATE QUESTION-ANSWER PAIRS BASED ON THE INFORMATION BELOW:
    - Please answer the following list of questions and provide the reasoning for each answer. 
    - Please format the response so that the reasoning is clearly separated from the answer. 
    - Place the reasoning section before the answer. 
    
    - Please quote direct full sentences of evidence from the report in the reasoning section to help justify the answer. 
    - Each question will provide the multiple options of the answer, pick one of them and follow the instructions on how to answer. 
    
    - An answer of "no" means that the report specifically confirms the answer to the question is no and there is clear evidence to confirm this. 
    - A reasoning of "inconclusive" means "insufficient conclusive evidence" or that there might be some evidence to indicate some answer, 
    but there isn't enough to confidently conclude an answer. 
    - An answer of "not discussed" means "not discussed in the report" or that the question topic was not mentioned in the report at all. 
    Keep the original numbering for the list of questions.
    
    - DO NOT include any questions that are not related to the "IMPRESSION" and "FINDINGS" sections.
    - DO NOT include any follow-up questions or questions that REQUIRE knowledge outside of the report
    - DO NOT include any questions that ask about 'residual' portions of the tumor or questions about a previous MRI
    - DO NOT use the phase 'in the report' in the questions. The questions should be answerable with only the MRI.
    - ALL ANSWERS should be in the text and should never be "None of the Above"
    
    RADIOLOGY REPORT:
    {row["note_text"]}

    """
    
    categorized = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": categorized_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Pairs",
                "schema": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "reasoning": {"type": "string"}
                                },
                            "required": ["question", "answer", "reasoning"]
                        }
                    }
                },
                "required": ["pairs"]
                }
            }
        },
        temperature=0.15
    )
    
    #Post-Processing 
    categorized_prompt2 = f"""
    Given a radiology report for a brain MRI, 
    Please go through each of the question-answer pairs and determine if the pairs are be answer given the criteria below. 
    
    Please answer the following list of questions and provide the reasoning for each answer. 
    Please format the response so that the reasoning is clearly separated from the answer. 
    Place the reasoning section before the answer. 
    
    Please quote direct full sentences of evidence from the report in the reasoning section to help justify the answer. 
    Each question will provide the multiple options of the answer, pick one of them and follow the instructions on how to answer.
    Keep the original numbering for the list of questions.
    
    If the question-answer pairs MEETS any of the criteria below, then tag them with the "NO" string.
    If the question-answer pairs does NOT MEET any of the criteria below, then tag them with the "YES" string.
    Also, be sure to explain why you chose the tag in the 'tag_reasoning' response.
    
    CRITERIA:
    - ANY question-answer pairs that require the patient's clinical history, previous brain MRIs, or any other information outside of the report to answer 
    (Questions about "midline shift" are ok and should NOT be tagged 'NO')
    - ANY question-answer pairs with the reasoning of "Inconclusive" and nothing else
    - ANY question-answer pairs with the answer of "Not discussed" and nothing else
    - ANY questions that explicitly asks to compare the MRI with a previous MRI or ask about a previous MRI. 
(Some keywords: postsurgical changes, postsurgical, retrospect, progression, recurrent, stable, tumor growth, tumor shrinkage, metastasis)
    - ANY questions that REQUIRE knowledge outside of the report to answer it.
    - ANY questions that ask about 'residual' portions of the tumor.
    - ANY question-answer pairs with the answer of "None of the above".
    - ANY question-answer pairs where it asks what technique is being used in the report in an explicit or implicit manner.
(Some keywords: multivoxel spectroscopy, FLAIR, T1, T2)
    - ANY questions that are not related to the brain  
    - ANY questions that are not about aspects found in the brain MRI
  
    Question-Answer Pairs:
    {categorized.choices[0].message.content}
    """
    
    
    categorized2 = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": categorized_prompt2}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Pairs",
                "schema": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "reasoning": {"type": "string"},
                                    "tag": {"type": "string"},
                                    "tag_reasoning": {"type": "string"}
                                },
                            "required": ["question", "answer", "reasoning", "tag", "tag_reasoning"]
                        }
                    }
                },
                "required": ["pairs"]
                }
            }
        },
        temperature=0.15
    )
    
    #Place the QA pairs into python data structures
    categories = json.loads(categorized2.choices[0].message.content)
    categories = categories['pairs']
    
    #Add all of the QA responses with their respective reports
    for entry in categories:
        skip = False #Variable used to determine if we need to skip the QA pair

        #exclude QA pairs with any of these words
        for keyword in ["postsurgical change", "Postsurgical change", "postsurgical changes", "Postsurgical changes", "postsurgical", "Postsurgical", "retrospect", "Retrospect", "progression", 
"Progression", "recurrent", "Recurrent", "stable", "Stable", "tumor growth", "Tumor growth", "tumor shrinkage", "retrospective", "Retrospective",
"Tumor shrinkage", "metastasis", "Metastasis", "spine", "Spine", "spinal", "Spinal", "\\*\\*\\*\\*\\*", "\\*\\*\\*\\*\\*'s", "metastases", "Metastases",
"diffus\\*", "perfus\\*", "flow", "MRA", "MRV", "metasta\\*", "contrast", "age", "date", "fiducial", "discussed", "specified", "purpose", "radiation", "susceptibility",
"Diffus\\*", "Perfus\\*", "Flow", "Metasta\\*", "Contrast", "Age", "Date", "Fiducial", "Discussed", "Specified", "Purpose", "Radiation", "Susceptibility"]:
            
            if re.search(rf"{keyword}", entry['question']) != None or re.search(rf"{keyword}", entry['answer']) != None:
                skip = True #Check if the keyword is found

        if skip: #If a keyword is found then do NOT add the QA pair
            continue

        #Add QAPair to the respective dictionary
        QAPairs['Accession Num'].append(row['accessionnumber'])
        QAPairs['Patient Pic ID'].append(row['patientepicid'])
        QAPairs['Patient Durable Key'].append(row['patientdurablekey'])
        QAPairs['DeID Note ID'].append(row['deid_note_id'])
        QAPairs['DeID Note CSN ID'].append(row['deid_note_csn_id'])
        QAPairs['Procedure ID'].append(row['procedureorderfactid'])
        QAPairs['DeID Note Key'].append(row['deid_note_key'])
        QAPairs['Original Note'].append(row['note_text'])
        QAPairs['Question'].append(entry['question'])
        QAPairs['Answer'].append(entry['answer'])
        QAPairs['Tag'].append(entry['tag'])
        
            
    print("Progress: ", round((idx+1-5000)/len(clinical_notes) * 100, 2), "%")

#Make the directory a dataframe for easy filtering
pairs = pd.DataFrame(QAPairs) 

#After creating and preprocessing all of the data points, filter out all QAPairs that are tagged 'NO'
pairs = pairs[pairs['Tag'] == 'YES']
pairs.drop(["Tag"], axis=1) #Drop the tag column since it is no longer needed
pairs.to_csv('ucsf_neuroimaging_pairs_5000_7000.csv') #save it to a .csv file
