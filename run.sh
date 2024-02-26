#!/bin/bash

# Replace the URL below with your repository URL
REPO_URL="https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency.git"
REPO_DIR="Sample_Capstone\code"

# Clone Git repository
git clone $REPO_URL
cd $REPO_DIR

# Install Python and pip
sudo apt-get update
sudo apt-get install -y python3
sudo apt-get install -y python3-pip

# Install required Python packages
pip3 install -r requirements.txt
pip install tensorflow

# Run main.py script
python3 main.py
