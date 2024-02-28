#!/bin/bash

# Replace the URL below with your repository URL
REPO_URL="https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency.git"
REPO_DIR="NN-Activation-Function-Computational-Efficiency/Sample_Capstone/code/"

# Clone Git repository
git clone $REPO_URL

# Install Python and pip
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -yq python3 python3-pip -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

pip3 install numpy
pip3 install pandas
pip3 install psutil
pip3 install scikit-learn
pip3 install tensorflow

# Run main.py script
python3 ${REPO_DIR}main.py "$@"
