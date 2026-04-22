from huggingface_hub import hf_hub_download

# Download a specific model file
model_name = "meta-llama/Llama-2-7b"
filename = "model.safetensors"

# Download to a local directory
file_path = hf_hub_download(
    repo_id=model_name,
    filename=filename,
    repo_type="model"
)

print(f"Downloaded to: {file_path}")