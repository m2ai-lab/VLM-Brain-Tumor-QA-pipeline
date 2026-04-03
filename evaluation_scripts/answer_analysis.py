import argparse
import pandas as pd
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer   
from pydantic import BaseModel, Field, ValidationError
import re 
import torch

SYSTEM_PROMPT = """
You are an expert AI behavior analyst. Your task is to analyze sets of questions that an AI model answered CORRECTLY versus INCORRECTLY. 
You must contrast the two sets and identify specific, actionable patterns that explain why the model fails on certain questions. Look for differences in:
1. Question complexity (e.g., multi-step reasoning vs. factual recall)
2. Structural formats (e.g., true/false, open-ended, medical jargon)
3. Specific keywords or themes unique to the failures.

Your response MUST be a strictly valid JSON object using the exact keys below. Do not nest the JSON inside any other objects. Do not include any markdown formatting outside the JSON block.

{
  "success_patterns": "Themes, topics, or structural patterns common ONLY in the questions the model got right.",
  "failure_patterns": "Themes, topics, or structural patterns common ONLY in the questions the model got wrong.",
  "key_differences": "The primary differences in wording, complexity, or subject matter between the right and wrong sets.",
  "insightful_conclusion": "A definitive statement on the model's blind spots and why it is failing."
}
"""

class QwenResponse(BaseModel):
    success_patterns: str = Field(description="Themes, topics, or structural patterns common ONLY in the questions the model got right.")
    failure_patterns: str = Field(description="Themes, topics, or structural patterns common ONLY in the questions the model got wrong.")
    key_differences: str = Field(description="The primary differences in wording, complexity, or subject matter between the right and wrong sets.")
    insightful_conclusion: str = Field(description="A definitive statement on the model's blind spots and why it is failing.")

def argument_handler():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--rights_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_right.csv")
    parser.add_argument('--wrongs_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/Top_wrong.csv")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/metrics/QA_Analysis.json")
    parser.add_argument('--model_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/Qwen2.5-7B-Instruct", help="Path or Hugging Face model name")
    
    return parser.parse_args()

def clean_json_string(raw_str):
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

def query_the_model(model, tokenizer, comparison_prompt, system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comparison_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate output with low temperature for analytical grounding
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        temperature=0.2, # Added temperature to make it more analytical and less "creative"
        do_sample=True
    ) 
    
    input_length = inputs['input_ids'].shape[1]
    new_tokens = generated_ids[0][input_length:]
    
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    cleaned_response = clean_json_string(output_text) 
    
    try:
        if not cleaned_response:
            raise ValueError("Regex failed to find anything resembling JSON.")
            
        validated_data = QwenResponse.model_validate_json(cleaned_response)
        return validated_data.model_dump()
        
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        return {"error": "Schema mismatch", "raw": output_text}
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return {"error": f"Parsing error: {str(e)}", "raw": output_text}

def main(args):

    # 1. Load the data 
    top_rights_df = pd.read_csv(args.rights_path)
    top_wrongs_df = pd.read_csv(args.wrongs_path)

    # 2. Clean data (Fixed: Don't strip by question mark, just strip whitespace)
    top_rights_df['Question'] = top_rights_df['Question'].astype(str).str.strip()
    top_wrongs_df['Question'] = top_wrongs_df['Question'].astype(str).str.strip()

    # 3. Extract questions with the highest counts
    most_right_rows = top_rights_df[top_rights_df['Count'] == top_rights_df['Count'].max()]
    most_wrong_rows = top_wrongs_df[top_wrongs_df['Count'] == top_wrongs_df['Count'].max()]

    # Extract strings
    all_right_str = ", ".join(sorted(set(top_rights_df['Question'].dropna())))
    all_wrong_str = ", ".join(sorted(set(top_wrongs_df['Question'].dropna())))
    most_right_str = ", ".join(sorted(set(most_right_rows['Question'].dropna())))
    most_wrong_str = ", ".join(sorted(set(most_wrong_rows['Question'].dropna())))

    # Initialize the model and tokenizer
    print(f"Loading Qwen from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 4. Create CONTRASTIVE testing frameworks
    # Instead of sending rights and wrongs separately, send them together.
    comparisons = {
        "Broad Analysis (All Rights vs All Wrongs)": f"SUCCESSFUL QUESTIONS:\n{all_right_str}\n\nFAILED QUESTIONS:\n{all_wrong_str}",
        "Extreme Analysis (Most Right vs Most Wrong)": f"MOST SUCCESSFUL QUESTIONS:\n{most_right_str}\n\nMOST FAILED QUESTIONS:\n{most_wrong_str}"
    }

    responses = {}

    for label, prompt_content in comparisons.items():
        print(f"Running Contrastive Analysis: {label}...")
        response = query_the_model(model, tokenizer, prompt_content, SYSTEM_PROMPT)
        responses[label] = response
        print(f"Completed {label}\n{'-'*30}")

    # 5. Save final analysis into JSON
    with open(args.output_path, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"Analysis saved to {args.output_path}")

if __name__ == "__main__":
    main(argument_handler())
