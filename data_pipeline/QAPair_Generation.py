# Python libraries
import os
import sys
import socket

#Third-party Libraries
import pandas as pd
import numpy as np
import openpyxl
from openai import AzureOpenAI
import json

#QA Generation Pipeline
def main():
    #grab the notes that were saved and convert them into a list for easy row access
    clinical_notes = pd.read_excel('bt_notes.xlsx')

    #creating the dictioanry that will hold the question answer pairs
    QAPairs = {'Note ID': [], 'Original Note': [], 'Question': [], 'Answer': [], 'Reasoning': [], 'Tag': []}

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
        - ALL ANSWERS should be in the text and should never be 
    
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
        tagged_prompt = f"""
        Given a radiology report for a pancreatic CT scan, 
        Please go through each of the question-answer pairs and determine if the pairs are be answer given the criteria below. 
    
        Please answer the following list of questions and provide the reasoning for each answer. 
        Please format the response so that the reasoning is clearly separated from the answer. 
        Place the reasoning section before the answer. 
    
        Please quote direct full sentences of evidence from the report in the reasoning section to help justify the answer. 
        Each question will provide the multiple options of the answer, pick one of them and follow the instructions on how to answer.
        Keep the original numbering for the list of questions.
    
        If the question-answer pairs MEETS any of the criteria below, then tag them with the "NO" string.
        If the question-answer pairs does NOT MEET any of the criteria below, then tag them with the "YES" string.
    
        CRITERIA:
        - A question-answer pair with the answer of "no" means that the report specifically confirms the answer to 
        the question is no and there is clear evidence to confirm this. 
    
        - A question-answer pair with the reasoning of "inconclusive" means "insufficient conclusive evidence" or that there
        might be some evidence to indicate some answer, but there isn't enough to confidently conclude an answer. 
    
        - A question-answer pair with the answer of "not discussed" means "not discussed in the report" or 
        that the question topic was not mentioned in the report at all. 
    
        - ANY questions that are not related to the "IMPRESSION" and "FINDINGS" sections.
        - ANY follow-up questions or questions that REQUIRE knowledge outside of the report
        - ANY questions that ask about 'residual' portions of the tumor or questions about a previous MRI
        - A question-answer pair with the answer of "None of the above"
 
  
        Question-Answer Pairs:
        {categorized.choices[0].message.content}
    
        RADIOLOGY REPORT:
        {row["note_text"]}
        """
    
    
        tagged = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": tagged_prompt}],
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
                                        "tag": {"type": "string"}
                                    },
                                "required": ["question", "answer", "reasoning", "tag"]
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
        categories = json.loads(tagged.choices[0].message.content)
        categories = categories['pairs']
    
    
        #Add all of the QA responses with their respective reports
        for entry in categories:
            #Add QAPair to the respective dictionary
            QAPairs['Note ID'].append(row['deid_note_key'])
            QAPairs['Original Note'].append(row['note_text'])
            QAPairs['Question'].append(entry['question'])
            QAPairs['Answer'].append(entry['answer'])
            QAPairs['Reasoning'].append(entry['reasoning'])
            QAPairs['Tag'].append(entry['tag'])
        
        #Print out status    
        print(idx, "out of ", len(clinical_notes))
    
    
#Run the Main function
if __name__ == "__main__":
    main()