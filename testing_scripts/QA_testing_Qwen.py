import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer   
import argparse
from pydantic import BaseModel, Field, ValidationError
import re 

# Added a strong system prompt to enforce JSON output
SYSTEM_PROMPT = """
You are an expert radiologist AI. You must output your response ONLY as a valid JSON object.
Do not include any conversational text before or after the JSON. 
Use the following format:
{
  "reasoning": "Your step-by-step clinical reasoning here.",
  "answer": "The final choice selected from the options."
}
"""

FEW_SHOT_EXAMPLE = """
Example Request:
Question: Based on the T2/FLAIR hyperintensity, what is the most likely grade? 1) Low Grade 2) High Grade

Example Response:
{
  "reasoning": "The slices show significant mass effect and central necrosis within the T1-contrast enhancing lesion, which is highly suggestive of aggressive growth.",
  "answer": "2) High Grade"
}
"""

class QwenResponse(BaseModel):
    reasoning: str = Field(description="Reasoning for answer")
    answer: str = Field(description="The final choice selected from the options.")

def clean_json_string(raw_str):
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

def query_the_model(model, tokenizer, question):
    # Use a system prompt to strongly enforce JSON
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{FEW_SHOT_EXAMPLE}\n====\nQuestion: {question}"}
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
    print(f"Loading Qwen from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # bf16 is great for newer models like Qwen to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    qa_data = pd.read_csv(args.qa_path)
    generated_answer = []
    generated_reasoning = []
    total = qa_data.shape[0]

    for idx, row in qa_data.iterrows():
        # Added a fallback in case "Assigned ID" isn't in your CSV
        row_id = row.get("Assigned ID", idx)
        print(f'Processing {idx+1}/{total} (ID: {row_id})...')
        
        # Passed tokenizer to the function
        response = query_the_model(model, tokenizer, row["Question"])
        
        generated_answer.append(response.get('answer', 'Error'))
        generated_reasoning.append(response.get('reasoning', 'Error'))
        print(f"Response: {response}\n{'-'*30}")
        
    # Save results
    qa_data["predicted_answer"] = generated_answer
    qa_data["Qwen_Reasoning"] = generated_reasoning
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    qa_data.to_csv(args.output_path, index=False)
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Text Inference")
    parser.add_argument('--qa_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")
    parser.add_argument('--output_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/Qwen/text_only_results.csv")
    parser.add_argument('--model_path', type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/Qwen2.5-7B-Instruct") 
    
    args = parser.parse_args()
    main(args)