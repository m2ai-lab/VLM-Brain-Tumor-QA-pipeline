#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

echo "Python 3 is installed: $(python3 --version)"

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi
echo "Virtual environment activated."

# ── Create gitignored data directories ───────────────────────────────────────
echo "Creating local data directories (models/ and QApairs/)..."
mkdir -p models QApairs
touch models/.gitkeep QApairs/.gitkeep
echo "  models/   ✓"
echo "  QApairs/  ✓"

# ── Bootstrap config ─────────────────────────────────────────────────────────
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo ""
    echo "  ⚠️  config.yaml created from template."
    echo "  Please open config.yaml and fill in your cluster paths before running experiments."
    echo ""
else
    echo "  config.yaml already exists — skipping."
fi

cd ..


echo cloning Med3DVLM repository...
if git clone git@github.com:mirthAI/Med3DVLM.git; then
    echo "Repository cloned successfully."
else if git clone https://github.com/mirthAI/Med3DVLM.git; then
    echo "Repository cloned successfully."
else
    echo "Repository already exists or cloning failed"
    exit 1 
fi

cd Med3DVLM || { echo "Failed to change directory to Med3DVLM"; exit 1; }


echo "Installing core packages..."
# Install core packages
pip install torch==2.6.0
pip install torchvision==0.21.0
pip install monai==1.4.0
pip install deepspeed==0.16.3
echo "Core packages installed."

# Install required packages
pip install -r requirements.txt
echo "Required packages installed."

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Setup completed successfully. You can now run the application using ./run.sh"
