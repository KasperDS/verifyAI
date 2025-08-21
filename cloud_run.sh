#!/bin/bash

# Basic setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv

# Clone your repo
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Force CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run your script (update this line to match your entrypoint)
python run_and_parse_wrapper.py
