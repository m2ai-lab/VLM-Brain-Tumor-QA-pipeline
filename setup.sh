#!/bin/bash

# =============================================================================
# setup.sh — Automated Environment and Model Setup
# =============================================================================

# Exit on error
set -e

# 1. Determine Project Root and Bootstrap Config
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
echo "Project root detected as: $PROJECT_ROOT"

echo "Starting project setup..."

# 1. Create base directories
mkdir -p models test_output logs environments/envs environments/requirements datasets/blank
echo "Directories created: models/, test_output/, logs/, environments/envs/, environments/requirements/"

# 2. Bootstrap config
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    # Update project_root in the newly created config.yaml
    sed -i "s|^project_root:.*|project_root: $PROJECT_ROOT|" config.yaml
    echo "✅ config.yaml created and project_root updated."
else
    echo "ℹ️ config.yaml already exists. Skipping bootstrap."
fi

# Helper function for Y/N prompts
confirm() {
    if [ "$YES_TO_ALL" = true ]; then
        return 0
    fi
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

# 2. Setup Mode Choice
YES_TO_ALL=false
echo "------------------------------------------------------------------"
echo "Welcome to the Brain MRI VQA Setup"
echo "------------------------------------------------------------------"
read -p "Enable 'Yes to All' mode? (Automatically approve all steps) [y/N]: " yto
case "$yto" in [yY][eE][sS]|[yY]) YES_TO_ALL=true ;; esac

if ! confirm "Do you want to run the automated setup? (Downloads/Environments)"; then
    echo "Setup cancelled by user."
    exit 0
fi

# 3. Create base directories
mkdir -p models QApairs logs datasets datasets/2D_slices datasets/blank environments/envs environments/requirements 
echo "Directories ensured: models/, QApairs/, logs/, datasets/, environments/envs/"

# 4. Create Bootstrap Environment (for data tools)
echo "--- Creating Bootstrap Environment (vlm-bootstrap) ---"
if [ ! -d "environments/envs/vlm-bootstrap" ]; then
    python3 -m venv environments/envs/vlm-bootstrap
    source environments/envs/vlm-bootstrap/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/bootstrap_reqs.txt
    deactivate
fi

# 5. Download External Datasets (Images, Metadata)
source environments/envs/vlm-bootstrap/bin/activate

if confirm "Download NIFTI Image datasets? (WARNING: 145GB)"; then
    echo "--- Starting Automated NIFTI Image Download ---"
    python utility_scripts/download_images.py
    NIFTI_DOWNLOADED=true
else
    echo "Skipping image download."
    NIFTI_DOWNLOADED=false
fi

if confirm "Download VQA dataset automatically?"; then
    echo "--- Starting Automated VQA Dataset Download ---"
    python utility_scripts/download_vqa_dataset.py
else
    echo "Skipping VQA dataset download."
fi

echo "--- Generating human QA subset ---"
python utility_scripts/make_human_dataset.py

echo "--- Generating reshuffled QA dataset ---"
python utility_scripts/make_reshuffle_dataset.py

echo "--- Generating blank images and saving to config.yaml ---"
python utility_scripts/make_blank_images.py

deactivate

# 6. Create Data Pipeline Environment (for slice extraction + montage generation)
echo "--- Creating Data Pipeline Environment (vlm-data-pipeline) ---"
if [ ! -d "environments/envs/vlm-data-pipeline" ]; then
    python3 -m venv environments/envs/vlm-data-pipeline
    source environments/envs/vlm-data-pipeline/bin/activate
    pip install --upgrade pip
    pip install -r environments/requirements/data_pipeline_reqs.txt
    deactivate
fi

# 7. Run slice extraction + montage generation (skip masks — requires GPU)
if [ "$NIFTI_DOWNLOADED" = true ] || confirm "Run slice extraction & montage generation now?"; then
    echo "--- Running generate_slices.py (--skip_masks, no GPU needed) ---"
    source environments/envs/vlm-data-pipeline/bin/activate
    python data_pipeline/generate_slices.py --skip_masks
    deactivate

    echo ""
    echo "=================================================================="
    echo "  NOTE: Tumour segmentation masks were NOT generated."
    echo "  Masks require a GPU and SwinUNETR weights."
    echo "  To generate masks on HPC, run:"
    echo ""
    echo "    sbatch slurm_scripts/sbatch_generate_masks"
    echo ""
    echo "  This will re-run generate_slices.py WITH mask generation."
    echo "=================================================================="
fi


# 6. Environment Creation (Model-specific)
if confirm "Create Model-specific Python environments and download weights?"; then
    echo "--- Creating Main Orchestrator Environment ---"
    if [ ! -d "environments/envs/vlm-orchestrator" ]; then
        python3 -m venv environments/envs/vlm-orchestrator
        source environments/envs/vlm-orchestrator/bin/activate
        pip install --upgrade pip
        pip install -r environments/requirements/orchestrator_reqs.txt
        deactivate
    fi

    if confirm "Download model weights now?"; then
        source environments/envs/vlm-orchestrator/bin/activate
        python utility_scripts/model_download.py
        deactivate
    fi

    echo "--- Setting up model-specific environments ---"
    
    # Function to setup venv
    setup_venv() {
        local name=$1
        local reqs=$2
        if [ ! -d "environments/envs/$name" ]; then
            echo "Setting up $name..."
            python3 -m venv "environments/envs/$name"
            source "environments/envs/$name/bin/activate"
            pip install --upgrade pip
            pip install -r "$reqs"
            deactivate
        fi
    }

    setup_venv "vlm-medgemma" "environments/requirements/medgemma_reqs.txt"
    setup_venv "vlm-lingshu" "environments/requirements/lingshu_reqs.txt"
    
    if [ ! -d "models/Med3DVLM" ]; then
        git clone https://github.com/mirthAI/Med3DVLM.git models/Med3DVLM
    fi
    setup_venv "vlm-med3dvlm" "environments/requirements/med3dvlm_reqs.txt"

    if [ ! -d "models/LLaVA-Med_Extension" ]; then
        git clone https://github.com/m2ai-lab/LLaVA-Med_Extension models/LLaVA-Med_Extension
    fi
    setup_venv "vlm-llavamed" "environments/requirements/llavamed_reqs.txt"

    setup_venv "vlm-medimageinsight" "environments/requirements/medimageinsight_reqs.txt"
    setup_venv "llm-qwen3" "environments/requirements/qwen3_reqs.txt"
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
    if [ ! -d "models/Med3DVLM" ]; then
        git clone https://github.com/mirthAI/Med3DVLM.git models/Med3DVLM
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
