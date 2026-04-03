import argparse
import pandas as pd
import os
from collections import defaultdict
import json
from transformers import AutoModelForCausalLM, AutoTokenizer   
from pydantic import BaseModel, Field, ValidationError
import re 
import torch


SYSTEM_PROMPT = """
Analyze the following questions and determine patterns in the types of questions specifying any common themes, topics, or formats. If not enough information is provided, state that explicitly. Your response should be a JSON object with the following format:
{
  "common_themes": "Description of any common themes or topics in the questions.",
  "common_formats": "Description of any common formats or structures in the questions.",
  "insufficient_information": "State if there is not enough information to determine patterns."
}
"""


class QwenResponse(BaseModel):
    common_themes: str = Field(description="Description of any common themes or topics in the questions.")
    common_formats: str = Field(description="Description of any common formats or structures in the questions.")
    insufficient_information: str = Field(description="State if there is not enough information to determine patterns.")


def argument_handler():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--rights_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_right.csv")
    parser.add_argument('--wrongs_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_wrong.csv")
    parser.add_argument('--answer_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/QA_Analysis.json")
    parser.add_argument('--model_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/Qwen2.5-7B-Instruct", help="Path or Hugging Face model name")
    
    return parser.parse_args()

def clean_json_string(raw_str):
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

def query_the_model(model, tokenizer, questions):
    # Use a system prompt to strongly enforce JSON
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Questions to analyze: {questions}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=1024) # Reduced from 32768 to save memory/time
    
    # CRITICAL FIX: Slice the output to ignore the input prompt tokens!
    input_length = inputs['input_ids'].shape[1]
    new_tokens = generated_ids[0][input_length:]
    
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean and parse the output
    cleaned_response = clean_json_string(output_text) 
    
    try:
        if not cleaned_response:
            raise ValueError("Regex failed to find anything resembling JSON.")
            
        # Validate against the Pydantic schema
        validated_data = QwenResponse.model_validate_json(cleaned_response)
        return validated_data.model_dump()
        
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"reasoning": "Schema mismatch", "answer": "Error", "raw": output_text}
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error", "raw": output_text}

def main(args):

    # 1. Load the data 
    top_rights_df = pd.read_csv(args.rights_path)
    top_wrongs_df = pd.read_csv(args.wrongs_path)

    if 'Question' not in top_rights_df.columns or 'Count' not in top_rights_df.columns:
        raise ValueError("rights CSV must contain 'Question' and 'Count' columns")
    if 'Question' not in top_wrongs_df.columns or 'Count' not in top_wrongs_df.columns:
        raise ValueError("wrongs CSV must contain 'Question' and 'Count' columns")

    # 2. Clean data to extract questions without the question mark and any trailing text
    top_rights_df['Question'] = top_rights_df['Question'].str.split('?').str[0]
    top_wrongs_df['Question'] = top_wrongs_df['Question'].str.split('?').str[0]

    # 3. Extract questions with the highest counts
    most_right_rows = top_rights_df[top_rights_df['Count'] == top_rights_df['Count'].max()]
    most_wrong_rows = top_wrongs_df[top_wrongs_df['Count'] == top_wrongs_df['Count'].max()]

    # Extract all questions 
    right_questions = sorted(set(top_rights_df['Question'].dropna()))
    wrong_questions = sorted(set(top_wrongs_df['Question'].dropna()))
    most_right_questions = sorted(set(most_right_rows['Question'].dropna()))
    most_wrong_questions = sorted(set(most_wrong_rows['Question'].dropna()))

    # Conglomerate all the questions into strings for input into LLM
    all_right_question_string = ", ".join(right_questions)
    all_wrong_question_string = ", ".join(wrong_questions)
    most_right_question_string = ", ".join(most_right_questions)
    most_wrong_question_string = ", ".join(most_wrong_questions)


    #Initialize the model and tokenizer
    print(f"Loading Qwen from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    #Create testing framework for the different question sets
    question_tests = {
        "All right Questions": all_right_question_string,
        "All wrong Questions": all_wrong_question_string,
        "Most right Questions": most_right_question_string,
        "Most wrong Questions": most_wrong_question_string
    }

    #Query the model for each question set and store responses in a dictionary
    responses = dict.fromkeys(question_tests.keys())
    for label, questions in question_tests.items():
        print(f"Analyzing {label}...")
        # Passed tokenizer to the function
        response = query_the_model(model, tokenizer, questions)
        responses[label] = response

        print(f"Response: {response}\n{'-'*30}")

    # 5. Save final analysis into JSON
    json_string = json.dumps(responses, indent=4)
    with open(args.output_path, "w") as f:
        f.write(json_string)


if __name__ == "__main__":
    main(argument_handler())
