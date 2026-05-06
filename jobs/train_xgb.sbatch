#!/bin/bash
#SBATCH --job-name=tp53-xgb
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/xgb_%j.out
#SBATCH --error=logs/xgb_%j.err

set -eo pipefail
cd "$HOME/tp53-mutation-predictor"
mkdir -p logs

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tp53-predictor

python src/train_xgb.py --n-genes 2000 --n-splits 5 --seed 42
