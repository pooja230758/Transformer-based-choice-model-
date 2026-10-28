#!/bin/bash
set -e

# Activate the virtual environment
source venv/bin/activate

# Install scikit-learn and torchvision since it's used in prepare_data.py and d2l
pip install scikit-learn torchvision

echo "Preparing data..."
cd Transformer
python prepare_data.py

echo "Training Transformer..."
python train_transformer.py

echo "Done!"
