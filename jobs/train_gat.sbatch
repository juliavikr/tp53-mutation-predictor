#!/bin/bash
#SBATCH --job-name=tp53-gat
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/gat_%j.out
#SBATCH --error=logs/gat_%j.err

set -eo pipefail
cd "$HOME/tp53-mutation-predictor"
mkdir -p logs data/processed

module load miniconda3 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tp53-predictor

: "${RUN_NAME:?missing}"
: "${GRAPH_FILE:?missing}"

echo "Run        : $RUN_NAME"
echo "Graph      : $GRAPH_FILE"
echo "Date       : $(date)"
echo "Host       : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train_gnn.py \
    --model-kind gat \
    --graph-file "data/processed/${GRAPH_FILE}" \
    --run-name "${RUN_NAME}" \
    --feature-set "${FEATURE_SET:-expr_zscore}" \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --n-layers "${N_LAYERS:-2}" \
    --gat-heads "${GAT_HEADS:-4}" \
    --dropout "${DROPOUT:-0.4}" \
    --batch-norm \
    --residual \
    --pos-weight "${POS_WEIGHT:-balanced}" \
    --max-epochs "${MAX_EPOCHS:-300}" \
    --patience "${PATIENCE:-30}" \
    --val-frac 0.15 \
    --batch-size 32 \
    --lr "${LR:-1e-3}" \
    --weight-decay 1e-4 \
    --seed 42

python src/make_plots.py
