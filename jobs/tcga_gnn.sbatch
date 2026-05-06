#!/bin/bash
#SBATCH --job-name=tp53-tcga-gnn
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/tcga_gnn_%j.out
#SBATCH --error=logs/tcga_gnn_%j.err

set -eo pipefail
cd "$HOME/tp53-mutation-predictor"
mkdir -p logs

module load miniconda3 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tp53-predictor

: "${RUN_NAME:?missing}"
: "${GRAPH_FILE:?missing}"
: "${MODEL_KIND:?missing}"

echo "Run        : $RUN_NAME"
echo "Graph      : $GRAPH_FILE"
echo "Model      : $MODEL_KIND"
echo "Date       : $(date)"
echo "Host       : $(hostname)"

python src/tcga_gnn_eval.py \
    --graph-file "data/processed/${GRAPH_FILE}" \
    --run-name "${RUN_NAME}" \
    --model-kind "${MODEL_KIND}" \
    --feature-set expr_zscore \
    --hidden-dim 128 \
    --n-layers "${N_LAYERS:-3}" \
    --gat-heads 4 \
    --dropout 0.4 \
    --max-epochs 200 \
    --patience 25 \
    --val-frac 0.15 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --seed 42
