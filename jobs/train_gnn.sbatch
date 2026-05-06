#!/bin/bash
#SBATCH --job-name=tp53-gcn
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/gcn_%j.out
#SBATCH --error=logs/gcn_%j.err

set -eo pipefail
cd "$HOME/tp53-mutation-predictor"
mkdir -p logs data/processed

module load miniconda3 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tp53-predictor

# Build the gene-gene co-expression graph if it doesn't exist yet.
if [[ ! -f data/processed/gene_graph.npz ]]; then
    python src/graph_construction.py --n-genes 2000 --threshold 0.5
fi

python src/train_gnn.py \
    --n-genes 2000 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --hidden-dim 64 \
    --dropout 0.5 \
    --seed 42

# Generate metrics table + plots once both models have produced OOF preds.
python src/make_plots.py
