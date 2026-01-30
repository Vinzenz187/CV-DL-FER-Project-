#!/bin/bash
#SBATCH --job-name=fer-training
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

set -e

cd ~/projects/my_pytorch_project/CV-DL-FER-Project-
source venv/bin/activate

python - << 'EOF'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

python -m src.training.training

