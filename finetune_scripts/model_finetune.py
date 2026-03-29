from transformers import Trainer, TrainingArguments, default_data_collator, AutoProcessor, AutoModelForImageTextToText
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from pydantic import BaseModel, Field, ValidationError
import re
import nibabel as nib
from PIL import Image
import os.path as path
from peft import LoraConfig, get_peft_model
from datasets import Dataset

IMAGE_DIR="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices"

MODEL_PATHS = {'Med3DVLM': "/mnt/fac/CX000019_DS1/UCSF-PGDM/PKG_-_UCSF-PDGM_Version_5/UCSF-PDGM-v5",
            'MedGemma' : "/scratch/group/CX000019_DS1/vlm-brain-mri/medgemma-1.5-4b-it"}

def clean_json_string(raw_str):
    # Remove markdown code blocks if present
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    # Extract only the content between the first { and last }
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

class MedResponse(BaseModel):
    # Literal ensures the answer MUST be one of these specific strings
    reasoning: str = Field(description="Step-by-step clinical observation of the MRI slices.")
    answer: str = Field(description="The final choice selected from the options.")

def process_slices(image_dir: str):
    """Loads specific PNG slices from a directory into a list of PIL Images."""
    slices = []
    for i in slice_names:
        slice_path = path.join(image_dir, f'{i}.png')
        print(slice_path)
        if path.exists(slice_path):
            # PIL requires Image.open() and it's best practice to ensure RGB format
            slices.append(Image.open(slice_path).convert("RGB"))
            print("slice loaded")
        else:
            print(f"Warning: {slice_path} not found.")
    
    return slices

def finetune (model_name: str, dataset):
    """
    The finetune function is used to properly train each model based on their
    own parts. For medgemma, we need a processor and the model. For Med3DVLM,
    it has a model and tokenizer.

    model_parts = (model, other parts (processor, tokenizer, etc.))
    dataset = QApairs used to fine tune (Should be a Dataframe)
    """
    #Split the dataset into train, validate, test 
    qa_train, qa_eval = train_test_split(dataset, train_size=0.7)

    if model_name == 'MedGemma':
        #Place all the related code for finetuning Med3DVLM
        #Set up the model
        model_id = MODEL_PATHS[model_name]
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )

        #With the model set up, we can now freeze all layers out side of the vision projector and LLM (set up LoRA for LLM)
        for param in model.parameters(): #Freeze all layers
            param.requires_grad = False

        for param in model.model.vision_tower.vision_model.parameters(): # unfreeze vision model and LLM
            param.requires_grad = True

        for param in model.model.language_model.parameters():
            param.requires_grad = True

        #Add LoRA for the LLM to make finetuning less computationally expensive
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05
        )
        model = get_peft_model(model, config)

        def qa_metrics(eval_pred):
            """
            This function is used to show how the prediction should be compared to the result.
            """
            #Get the predictions based on the generated output
            predictions, labels = eval_pred
            print(predictions)
            prediction = np.argmax(predictions, axis=-1)

            print(prediction)

            decoded_preds = processor.decode(prediction, skip_special_tokens=False).strip()
            decoded_labels = processor.decode(labels, skip_special_tokens=True).strip()

            print(decoded_preds)
            print(decoded_labels)

            correct = 0
            total = len(decoded_preds)

            for pred, label in zip(decoded_preds, decoded_labels):
                # Since we pre-filled '{', the model output likely starts with "reasoning": ...
                full_json_str = "{" + raw_response 
                if not full_json_str.endswith("}"):
                    full_json_str += "}"

                response = clean_json_string(pred) 
                label = label.strip()

                print(response)
                print(label)

                # Extract only the answer portion
                try:
                    # Extract JSON string using regex
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if not json_match:
                        raise ValueError("No JSON found in model output")

                    # Validate against the Pydantic schema
                    validated_data = MedResponse.model_validate_json(json_match.group(0))

                    response = validated_data.model_dump()
                    print(validated_data)

                except ValidationError as e:
                    print(f"Pydantic Validation Error: {e}")
                    return {"reasoning": "Schema mismatch", "answer": "Error", "raw": cleaned_response}
                except Exception as e:
                    return {"reasoning": f"Parsing error: {str(e)}", "answer": "Error"}

                if label in response['answer']:
                    correct += 1

            print(round(float(correct / total), 2), "%")
            return {"exact_match": float(correct / total)}

        class BrainCollator:
            """
            This is a data collator used to transform the inputs into of each entry into
            a prompt that the LLM can respond to.
            """
            def __init__(self, processor):
                self.processor = processor

            def __call__(self, batch):
                #create lists for both the prompts and images (idecies map images to prompts)
                prompts = []
                imgs = []

                print("doing prompt creation")
                print(len(batch))
                for entry in batch:
                     # Assume the PNGs are stored in a folder named after the patient ID
                    patient_image_dir = path.join(IMAGE_DIR, str(entry["Assigned ID"]))
                    print(patient_image_dir)
                    if not path.exists(patient_image_dir):
                        return {"reasoning": f"Error: Directory {patient_image_dir} not found.", "answer": "Error"}

                    # 1. Get processed PIL images
                    images = process_slices(patient_image_dir)
                    num_loaded_slices = len(images)
                    print(num_loaded_slices)

                    if num_loaded_slices == 0:
                         return {"reasoning": "Error: No slices found to process.", "answer": "Error"}
                    #if there are images then append to images
                    imgs.append(images)

                    # More aggressive prompt with a clear JSON schema
                    print("Creating prompt text")
                    prompt_text = (
                        "Instruction: You are a neuroradiologist. Analyze the MRI slices and provide a structured JSON response.\n"
                        f"{FEW_SHOT_EXAMPLE}"
                        "---\n"
                        f"Actual Question: {entry['Question']}\n"
                        f"Response: {entry['Answer']}"
                    )
                    prompts.append(prompt_text)
                print("prompts all made")
                # Build messages
                content = [{"type": "image"}] * 3
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]

                # Use add_generation_prompt=False because we are manually adding the "{"
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                print("Template loaded")
                # Encode multimodal input
                inputs = processor(
                    text=prompts,
                    images=images,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device, dtype=model.dtype)

                # Causal LM setup
                inputs["labels"] = inputs["input_ids"].copy()
                print("Done")
                return inputs

        #convert to Dataset objects then preprocess 
        #(only grab necessary information, such as question, answer, image_path)
        qa_train = qa_train[["Assigned ID", "Question", "Answer"]]
        qa_eval = qa_eval[["Assigned ID", "Question", "Answer"]]
        qa_train = Dataset.from_pandas(qa_train)
        qa_eval = Dataset.from_pandas(qa_eval)

        #Initialize the training arguments for the Trainer object
        training_args = TrainingArguments(
                output_dir="./brain_qa_evaluater",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                eval_strategy="steps",
                eval_steps=500,
                logging_steps=100,
                save_steps=500,
                num_train_epochs=3,
                learning_rate=2e-5,
                fp16=True,
                report_to="none",
                remove_unused_columns=False,
                load_best_model_at_end=True
            )


        ##Set up the trainer object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=qa_train,
            eval_dataset=qa_eval,
            data_collator=BrainCollator(processor),
            compute_metrics=qa_metrics,
        )

        #Run the model trainer and then save it once finetuned
        trainer.train()
        trainer.save_model("/scratch/group/CX000019_DS1/vlm-brain-mri/medgemma-finetuned")


def main():
    model='MedGemma'
    data = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")
    
    finetune(model, data)

if __name__ == "__main__":
    main()