from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda")  

dtype = torch.bfloat16 

model_id = "/scratch/group/CX000019_DS1/vlm-brain-mri/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B"
    
print(f"Loading Med3DVLM from {model_id}...")
# Using bfloat16 is highly recommended for MedGemma if your GPU supports it
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

for name, module in model.named_modules():
    print(name)

for name, param in model.named_parameters():
    print(name)