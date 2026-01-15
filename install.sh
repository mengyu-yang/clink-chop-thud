#!/bin/bash
# Installation script for Clink! Chop! Thud!
# Usage: bash install.sh [env_name]

set -e

ENV_NAME="${1:-clink_chop_thud}"

conda create -n ${ENV_NAME} python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
pip install -r requirements.txt

echo "Done. Activate with: conda activate ${ENV_NAME}"
