from transformers import LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch

# Using bfloat16 is highly recommended for MedGemma if your GPU supports it
model_ids = ["/scratch/group/CX000019_DS1/vlm-brain-mri/medgemma-1.5-4b-it", "/scratch/group/CX000019_DS1/vlm-brain-mri/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B", "/scratch/group/CX000019_DS1/vlm-brain-mri/LLaVA-Med/llava-med-v1.5-mistral-7b"]

#Medgemma = 0
#Med3DVLM = 1
#LLaVA-Med = 2

model_id = model_ids[1]
                
device = "cuda" if torch.cuda.is_available() else "cpu"

#Based on what we want to inspect, load the model
if model_id == "/scratch/group/CX000019_DS1/vlm-brain-mri/medgemma-1.5-4b-it":
    print(f"Loading MedGemma from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
elif model_id == "/scratch/group/CX000019_DS1/vlm-brain-mri/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B":
    max_length = 1024
    image_size = (128, 256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).to(device=device)
else:
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )  

#Print out the modules and trainable parameters
print("Modules: \n")
for name, module in model.named_modules():
    print(name)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")