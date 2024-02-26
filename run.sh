#!/bin/bash

# Replace the URL below with your repository URL
REPO_URL="https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency.git"
REPO_DIR="NN-Activation-Function-Computational-Efficiency/Sample_Capstone/code/"

# Clone Git repository
git clone $REPO_URL
cd $REPO_DIR

# Install Python and pip
sudo apt-get update
sudo apt-get install -yq python3 python3-pip

# Install required Python packages
pip install numpy
pip install pandas
pip install psutil
pip install scikit-learn
pip install tensorflow

# Run main.py script
python3 main.py
