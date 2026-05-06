#!/bin/bash

# =============================================================================
# setup.sh — Automated Environment and Model Setup
# =============================================================================

# Exit on error
set -e
# 1. Determine Project Root and Bootstrap Config
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
echo "Project root detected as: $PROJECT_ROOT"

if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    # Update project_root in the newly created config.yaml
    # Use | as delimiter in sed to handle slashes in paths
    sed -i "s|^project_root:.*|project_root: $PROJECT_ROOT|" config.yaml
    echo "✅ config.yaml created and project_root updated."
else
    echo "ℹ️ config.yaml already exists. Skipping bootstrap."
fi

# Helper function for Y/N prompts
confirm() {
    local prompt="$1"
    read -p "$prompt [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# 2. Create base directories
mkdir -p models QApairs logs datasets environments/envs environments/requirements
echo "Directories ensured: models/, QApairs/, logs/, datasets/, environments/envs/"

# 3. Download External Datasets (UCSF-PDGM Images, Metadata)
if confirm "Do you want to download the UCSF-PDGM datasets automatically?"; then
    echo "--- Starting Automated UCSF-PDGM Image Download ---"
    python3 utility_scripts/download_ucsf_pdgm.py
    echo "Images and metadata pulled into datasets/ folder."
else
    echo "Skipping dataset download. Please ensure files are in datasets/ or update config.yaml."
fi

# 4. Create Main Orchestrator Environment
if confirm "Do you want to create the main Python environments?"; then
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

    # 5. Download Models
    if confirm "Do you want to download the model weights now?"; then
        echo "--- Downloading Models ---"
        source environments/envs/vlm-orchestrator/bin/activate
        python utility_scripts/model_download.py
        deactivate
        echo "Models downloaded to models/"
    else
        echo "Skipping model download. You will need to provide weights manually in models/."
    fi

    # 6. Create Individual Environments for Models
    echo "--- Setting up model-specific environments ---"
    
    # MedGemma
    if [ ! -d "environments/envs/vlm-medgemma" ]; then
        python3 -m venv environments/envs/vlm-medgemma
        source environments/envs/vlm-medgemma/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/medgemma_reqs.txt
        deactivate
    fi

    # Lingshu-32B
    if [ ! -d "environments/envs/vlm-lingshu" ]; then
        python3 -m venv environments/envs/vlm-lingshu
        source environments/envs/vlm-lingshu/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/lingshu_reqs.txt
        deactivate
    fi

    # Med3DVLM
    if [ ! -d "environments/envs/vlm-med3dvlm" ]; then
        if [ ! -d "models/Med3DVLM" ]; then
            git clone https://github.com/mirthAI/Med3DVLM.git models/Med3DVLM
        fi
        python3 -m venv environments/envs/vlm-med3dvlm
        source environments/envs/vlm-med3dvlm/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/med3dvlm_reqs.txt
        deactivate
    fi

    # LLaVA-Med
    if [ ! -d "environments/envs/vlm-llavamed" ]; then
        if [ ! -d "models/LLaVA-Med_Extension" ]; then
            git clone https://github.com/m2ai-lab/LLaVA-Med_Extension models/LLaVA-Med_Extension
        fi
        python3 -m venv environments/envs/vlm-llavamed
        source environments/envs/vlm-llavamed/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/llavamed_reqs.txt
        deactivate
    fi

    # MedImageInsight
    if [ ! -d "environments/envs/vlm-medimageinsight" ]; then
        python3 -m venv environments/envs/vlm-medimageinsight
        source environments/envs/vlm-medimageinsight/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/medimageinsight_reqs.txt
        deactivate
    fi
else
    echo "Skipping environment creation."
fi

echo "=================================================================="
echo "Setup complete!"
echo "Check config.yaml to ensure all paths are correct for your system."
echo "=================================================================="
