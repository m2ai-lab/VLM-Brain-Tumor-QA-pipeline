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
if [ ! -d "environments/envs/vlm-orchestrator" ]; then
    python3 -m venv environments/envs/vlm-orchestrator
    source environments/envs/vlm-orchestrator/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/orchestrator_reqs.txt
    echo "Main environment ready."
    deactivate
else
    echo "environments/envs/vlm-orchestrator already exists. Skipping..."
fi

# 4. Download Models
echo "--- Downloading Models ---"
# We run this using the orchestrator environment
source environments/envs/vlm-orchestrator/bin/activate
python utility_scripts/model_download.py
deactivate
echo "Models downloaded to models/"

# 5. Create Individual Environments for Models
# We use venv for each to ensure isolation. 

# -- MedGemma Environment --
echo "--- Creating MedGemma Environment (vlm-medgemma) ---"
if [ ! -d "environments/envs/vlm-medgemma" ]; then
    python3 -m venv environments/envs/vlm-medgemma
    source environments/envs/vlm-medgemma/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/medgemma_reqs.txt
    echo "MedGemma environment ready."
    deactivate
else
    echo "environments/envs/vlm-medgemma already exists. Skipping..."
fi

# -- Lingshu-32B Environment --
echo "--- Creating Lingshu Environment (vlm-lingshu) ---"
if [ ! -d "environments/envs/vlm-lingshu" ]; then
    python3 -m venv environments/envs/vlm-lingshu
    source environments/envs/vlm-lingshu/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/lingshu_reqs.txt
    echo "Lingshu environment ready."
    deactivate
else
    echo "environments/envs/vlm-lingshu already exists. Skipping..."
fi

# -- Med3DVLM Environment --
echo "--- Creating Med3DVLM Environment (vlm-med3dvlm) ---"
if [ ! -d "environments/envs/vlm-med3dvlm" ]; then
    if [ ! -d "Med3DVLM" ]; then
        git clone https://github.com/mirthAI/Med3DVLM.git
    fi
    python3 -m venv environments/envs/vlm-med3dvlm
    source environments/envs/vlm-med3dvlm/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/med3dvlm_reqs.txt
    echo "Med3DVLM environment ready."
    deactivate
else
    echo "environments/envs/vlm-med3dvlm already exists. Skipping..."
fi

# -- LLaVA-Med Environment --
echo "--- Creating LLaVA-Med Environment (vlm-llavamed) ---"
if [ ! -d "environments/envs/vlm-llavamed" ]; then
    python3 -m venv environments/envs/vlm-llavamed
    source environments/envs/vlm-llavamed/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/llavamed_reqs.txt
    echo "LLaVA-Med environment ready."
    deactivate
else
    echo "environments/envs/vlm-llavamed already exists. Skipping..."
fi

# -- MedImageInsight Environment --
echo "--- Creating MedImageInsight Environment (vlm-medimageinsight) ---"
if [ ! -d "environments/envs/vlm-medimageinsight" ]; then
    python3 -m venv environments/envs/vlm-medimageinsight
    source environments/envs/vlm-medimageinsight/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/medimageinsight_reqs.txt
    echo "MedImageInsight environment ready."
    deactivate
else
    echo "environments/envs/vlm-medimageinsight already exists. Skipping..."
fi

echo "=================================================================="
echo "Setup complete!"
echo "Environments created in environments/envs/"
echo "Models downloaded in models/"
echo ""
echo "To run experiments, use the main orchestrator environment:"
echo "source environments/envs/vlm-orchestrator/bin/activate"
echo "python experiment_orchestrator/run_experiments.py ..."
echo "=================================================================="
