from transformers import Trainer, TrainingArguments, default_data_collator, AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, LlavaForConditionalGeneration
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from pydantic import BaseModel, Field, ValidationError
import re
import nibabel as nib
from PIL import Image
import os.path as path
import os
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import logging
import SimpleITK as sitk
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_DIR="/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices"
NIFTI_DIR="/mnt/fac/CX000019_DS1/UCSF-PGDM/PKG_-_UCSF-PDGM_Version_5/UCSF-PDGM-v5"

MODEL_PATHS = {
    'Med3DVLM': "/scratch/group/CX000019_DS1/vlm-brain-mri/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B",
    'MedGemma' : "/scratch/group/CX000019_DS1/vlm-brain-mri/medgemma-1.5-4b-it",
    'LlavaMed' : "/scratch/group/CX000019_DS1/vlm-brain-mri/LLaVA-Med/llava-med-v1.5-mistral-7b"
}

SLICE_NAMES = ["Axial", "Coronal", "Sagittal"]

FEW_SHOT_EXAMPLE = """
Example Request:
Question: Based on the T2/FLAIR hyperintensity, what is the most likely grade? 1) Low Grade 2) High Grade

Example Response:
{
  "reasoning": "The slices show significant mass effect and central necrosis within the T1-contrast enhancing lesion, which is highly suggestive of aggressive growth.",
  "answer": "2) High Grade"
}
"""

def clean_json_string(raw_str):
    clean_str = re.sub(r'```json|```', '', raw_str).strip()
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    return match.group(0) if match else clean_str

class MedResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step clinical observation of the MRI slices.")
    answer: str = Field(description="The final choice selected from the options.")

def process_slices(image_dir: str):
    """Loads specific PNG slices from a directory into a list of PIL Images."""
    slices = []
    for i in SLICE_NAMES:
        slice_path = path.join(image_dir, f'{i}.png')
        if path.exists(slice_path):
            slices.append(Image.open(slice_path).convert("RGB"))
        else:
            print(f"Warning: {slice_path} not found.")
    return slices

# ======================= COLLATORS =======================

class MedGemmaCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        prompts = []
        imgs_list = []

        for entry in batch:
            patient_image_dir = path.join(IMAGE_DIR, str(entry["Assigned ID"]))
            if not path.exists(patient_image_dir):
                return {"reasoning": f"Error: Directory {patient_image_dir} not found.", "answer": "Error"}

            images = process_slices(patient_image_dir)
            if len(images) == 0:
                 return {"reasoning": "Error: No slices found.", "answer": "Error"}
            imgs_list.append(images)

            prompt_text = (
                "Instruction: You are a neuroradiologist. Analyze the MRI slices and provide a structured JSON response.\n"
                f"{FEW_SHOT_EXAMPLE}"
                "---\n"
                f"Actual Question: {entry['Question']}\n"
            )
            prompts.append(prompt_text)

        # Build messages for the batch
        # We only support batch size 1 natively here to avoid complex padding maps
        entry = batch[0]
        images = imgs_list[0]
        
        content = [{"type": "image"}] * 3 
        content.append({"type": "text", "text": prompts[0]})
        asst_content = [{"type": "text", "text": f"Actual Answer: {entry['Answer']}"}]
        messages = [{"role": "user", "content": content}, {"role": "assistant", "content": asst_content}]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        inputs = self.processor(
            text=input_text,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        try:
            image_token_id = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])]
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id[0]] = -100
            labels[labels == 262144] = -100 # MedGemma specific ignore token
        except Exception:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


class LlavaMedCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        entry = batch[0]
        patient_image_dir = path.join(IMAGE_DIR, str(entry["Assigned ID"]))
        images = process_slices(patient_image_dir)
        
        prompt_text = (
            "Instruction: You are a neuroradiologist. Analyze the MRI slices and return ONLY valid JSON.\n"
            f"{FEW_SHOT_EXAMPLE}\n"
            "---\n"
            f"Actual Question: {entry['Question']}\n"
        )
        
        content = [{"type": "image"}] * len(images)
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": f"Actual Answer: {entry['Answer']}"}]}]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        
        inputs = self.processor(
            text=input_text,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


class Med3DCollator:
    def __init__(self, tokenizer, proj_out_num=256):
        self.tokenizer = tokenizer
        self.proj_out_num = proj_out_num

    def __call__(self, batch):
        entry = batch[0]
        patient_id = entry["Assigned ID"]
        full_image_path = path.join(NIFTI_DIR, f'{patient_id}_nifti', f'{patient_id}_FLAIR.nii.gz')
        
        if not path.exists(full_image_path):
            raise FileNotFoundError(f"Missing {full_image_path}")
            
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(full_image_path))
        image_pt = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        
        target_size = (128, 256, 256)
        image_pt = F.interpolate(image_pt, size=target_size, mode='trilinear', align_corners=False).squeeze(0) # (1, D, H, W)

        image_tokens = "<im_patch>" * self.proj_out_num
        input_txt = image_tokens + entry["Question"] + f"\nActual Answer: {entry['Answer']}"
        
        inputs = self.tokenizer(input_txt, return_tensors="pt", max_length=512, truncation=True)
        inputs["images"] = image_pt 
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        return inputs

# ======================= FINETUNE =======================

def finetune(model_name: str, dataset):
    """
    Unified Finetuning dispatcher.
    """
    qa_train, qa_eval = train_test_split(dataset, train_size=0.7)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = MODEL_PATHS.get(model_name)
    if not model_id:
         raise ValueError(f"Model ID not found for {model_name}")

    if model_name == 'MedGemma':
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        for param in model.parameters(): param.requires_grad = False
        for param in model.model.vision_tower.vision_model.parameters(): param.requires_grad = True
        for param in model.model.language_model.parameters(): param.requires_grad = True

        config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
        model = get_peft_model(model, config)
        collator = MedGemmaCollator(processor)

    elif model_name == 'LlavaMed':
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        for param in model.parameters(): param.requires_grad = False
        for param in model.language_model.parameters(): param.requires_grad = True

        config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
        model = get_peft_model(model, config)
        collator = LlavaMedCollator(processor)

    elif model_name == 'Med3DVLM':
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512, padding_side="right", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )

        for param in model.parameters(): param.requires_grad = False
        # Unfreeze vision projector and LLM
        for name, param in model.named_parameters():
             if 'embed' in name or 'proj' in name: param.requires_grad = True

        config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
        model = get_peft_model(model, config)
        collator = Med3DCollator(tokenizer, proj_out_num=getattr(model.get_model().config, "proj_out_num", 256))

    else:
        raise ValueError(f"Model {model_name} not supported.")

    def qa_metrics(eval_pred):
        # Base metric function; exact evaluation handles decoding and Regex matching.
        return {"exact_match": 0.0}

    qa_train = qa_train[["Assigned ID", "Question", "Answer"]]
    qa_eval = qa_eval[["Assigned ID", "Question", "Answer"]]
    dataset_train = Dataset.from_pandas(qa_train)
    dataset_eval = Dataset.from_pandas(qa_eval)

    training_args = TrainingArguments(
            output_dir=f"/scratch/group/CX000019_DS1/vlm-brain-mri/{model_name}-finetuned",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=10,
            log_level="info",
            save_steps=500,
            num_train_epochs=3,
            learning_rate=2e-5,
            fp16=(model_name != 'Med3DVLM'),
            bf16=(model_name == 'Med3DVLM'), # Med3DVLM uses bf16
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=collator,
        compute_metrics=qa_metrics,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


def main():
    # You can switch this target manually: 'MedGemma', 'LlavaMed', 'Med3DVLM'
    model='MedGemma' 
    data = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/updated_ucsf_pdgm_pairs.csv")
    data = data.iloc[:100]
    
    finetune(model, data)

if __name__ == "__main__":
    main()