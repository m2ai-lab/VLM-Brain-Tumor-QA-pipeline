#!/bin/bash

# =============================================================================
# setup.sh — Automated Environment and Model Setup
# =============================================================================

# Exit on error
set -e

echo "Starting project setup..."

# 1. Create base directories
mkdir -p models QApairs logs envs
echo "Directories created: models/, QApairs/, logs/"

# 2. Bootstrap config
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo "⚠️ config.yaml created from template. Please update it with your paths."
fi

# 3. Create Main Orchestrator Environment (venv)
echo "--- Creating Main Orchestrator Environment (vlm-orchestrator) ---"
python3 -m venv envs/vlm-orchestrator
source envs/vlm-orchestrator/bin/activate
pip install --upgrade pip
pip install pandas pyyaml huggingface_hub pydantic
echo "Main environment ready."

# 4. Download Models
echo "--- Downloading Models ---"
# We run this using the orchestrator environment
python utility_scripts/model_download.py
echo "Models downloaded to models/"

# 5. Create Individual Environments for Models
# We use venv for each to ensure isolation. 

# -- MedGemma Environment --
echo "--- Creating MedGemma Environment (vlm-medgemma) ---"
python3 -m venv envs/vlm-medgemma
source envs/vlm-medgemma/bin/activate
pip install torch torchvision monai transformers accelerate pillow pandas pydantic
echo "MedGemma environment ready."
deactivate

# -- Lingshu-32B Environment --
echo "--- Creating Lingshu Environment (vlm-lingshu) ---"
python3 -m venv envs/vlm-lingshu
source envs/vlm-lingshu/bin/activate
pip install torch torchvision transformers accelerate qwen-vl-utils pillow pandas pydantic
echo "Lingshu environment ready."
deactivate

# -- Med3DVLM Environment --
# Note: Med3DVLM often has complex requirements, so we clone and install its requirements.txt
echo "--- Creating Med3DVLM Environment (vlm-med3dvlm) ---"
if [ ! -d "Med3DVLM" ]; then
    git clone https://github.com/mirthAI/Med3DVLM.git
fi
python3 -m venv envs/vlm-med3dvlm
source envs/vlm-med3dvlm/bin/activate
pip install -r Med3DVLM/requirements.txt
pip install deepspeed monai # Additional core packages from previous setup
echo "Med3DVLM environment ready."
deactivate

# -- LLaVA-Med Environment --
echo "--- Creating LLaVA-Med Environment (vlm-llavamed) ---"
python3 -m venv envs/vlm-llavamed
source envs/vlm-llavamed/bin/activate
pip install torch torchvision transformers accelerate pillow pandas pydantic
echo "LLaVA-Med environment ready."
deactivate

# -- MedImageInsight Environment --
echo "--- Creating MedImageInsight Environment (vlm-medimageinsight) ---"
python3 -m venv envs/vlm-medimageinsight
source envs/vlm-medimageinsight/bin/activate
# Standard CLIP/Transformer requirements
pip install torch torchvision transformers pillow pandas pydantic
echo "MedImageInsight environment ready."
deactivate

echo "=================================================================="
echo "Setup complete!"
echo "Environments created in envs/"
echo "Models downloaded in models/"
echo ""
echo "To run experiments, use the main orchestrator environment:"
echo "source envs/vlm-orchestrator/bin/activate"
echo "python experiment_orchestrator/run_experiments.py ..."
echo "=================================================================="
