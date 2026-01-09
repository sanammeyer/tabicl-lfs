#!/bin/bash

# SLURM job: PDLC-head fine-tuning for mini-TabICL Stage-1
# - Uses withheld Stage-1 priors (indices >=100000)
# - Starts from the Stage-2 SA checkpoint and adapts TFicl + PDLC head

# SLURM Directives
# Adjust cluster/partition if needed for your environment.
#SBATCH -J mini-TabICL_S1_PDL_FT      # Job name
#SBATCH -o ./%x.%j.%N.out             # STDOUT log
#SBATCH -e ./%x.%j.%N.err             # STDERR log
#SBATCH --get-user-env                # Load user env
#SBATCH --export=NONE                 # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std          # Partition/queue
#SBATCH --nodes=1                     # Single node
#SBATCH --gpus-per-node=4             # GPUs per node (A100 80GB x4)
#SBATCH --ntasks=1                    # One task
#SBATCH --cpus-per-task=36            # CPU cores
#SBATCH --mem=128G                    # RAM
#SBATCH --time=168:00:00              # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths (hard-coded, adjust if needed) ---
PROJECT_ROOT="/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs"
VENV_ACTIVATE="$PROJECT_ROOT/.tabicl/bin/activate"
PY_SRC_DIR="$PROJECT_ROOT/src"

# Withheld Stage-1 priors (we start from index 100000)
PRIOR_DIR="/dss/lxclscratch/0E/ra63pux2/ra63pux2/tabicl_s1_priors"
LOAD_PRIOR_START=100000

# Base checkpoint: mini-TabICL Stage-2 SA (self-attention)
BASE_CKPT_PATH="$PROJECT_ROOT/checkpoints/mini_tabicl_stage2_sa/step-1000.ckpt"
# Fine-tuning checkpoint directory (used by both phases)
CKPT_DIR="$PROJECT_ROOT/checkpoints/mini_tabicl_stage1_pdlc"

cd "$PROJECT_ROOT"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "[ERROR] Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

export PYTHONPATH="$PY_SRC_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

mkdir -p "$CKPT_DIR" || true
mkdir -p "$PROJECT_ROOT/training_metrics" || true

# --- Sanity checks ---
if [[ ! -d "$PRIOR_DIR" ]]; then
  echo "[ERROR] PRIOR_DIR does not exist: $PRIOR_DIR" >&2
  exit 1
fi
if [[ ! -r "$PRIOR_DIR" ]]; then
  echo "[ERROR] PRIOR_DIR is not readable: $PRIOR_DIR" >&2
  exit 1
fi
BATCH_COUNT=$(find "$PRIOR_DIR" -maxdepth 1 -type f -name 'batch_*.pt' | wc -l | awk '{print $1}')
if [[ "${BATCH_COUNT:-0}" -eq 0 ]]; then
  echo "[ERROR] PRIOR_DIR contains no batch_*.pt files: $PRIOR_DIR" >&2
  exit 1
fi
if [[ ! -r "$PRIOR_DIR/metadata.json" ]]; then
  echo "[WARN] metadata.json not found in PRIOR_DIR; proceeding anyway" >&2
fi
if [[ ! -f "$BASE_CKPT_PATH" ]]; then
  echo "[ERROR] Base checkpoint not found: $BASE_CKPT_PATH" >&2
  exit 1
fi

echo "[INFO] Using Stage-1 priors from:   $PRIOR_DIR (files: $BATCH_COUNT, start_index=$LOAD_PRIOR_START)"
echo "[INFO] Base Stage-2 SA checkpoint:  $BASE_CKPT_PATH"
echo "[INFO] Fine-tune checkpoints will be saved to: $CKPT_DIR"

TRAIN_MODULE="tabicl.train.run"

echo "[INFO] ===== PDLC fine-tuning on withheld Stage-1 priors (TFcol/TFrow frozen, TFicl+PDLC head trainable) ====="

srun torchrun --standalone --nproc_per_node=4 \
  -m "$TRAIN_MODULE" \
  --wandb_log False \
  --wandb_project mini-TabICL \
  --wandb_name mini-S1-PDL-Finetune \
  --wandb_dir "$PROJECT_ROOT/wandb" \
  --wandb_mode disabled \
  --device cuda \
  --dtype float32 \
  --amp True \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 2000 \
  --batch_size 512 \
  --micro_batch_size 4 \
  --lr 2e-5 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.1 \
  --gradient_clipping 1.0 \
  --prior_dir "$PRIOR_DIR" \
  --load_prior_start "$LOAD_PRIOR_START" \
  --delete_after_load False \
  --prior_device cpu \
  --checkpoint_path "$BASE_CKPT_PATH" \
  --only_load_model True \
  --embed_dim 128 \
  --col_num_blocks 3 \
  --col_nhead 4 \
  --col_num_inds 128 \
  --row_num_blocks 3 \
  --row_nhead 8 \
  --row_num_cls 4 \
  --row_rope_base 100000 \
  --icl_num_blocks 12 \
  --icl_nhead 4 \
  --ff_factor 2 \
  --norm_first True \
  --freeze_col True \
  --freeze_row True \
  --freeze_icl False \
  --row_elliptical False \
  --icl_elliptical False \
  --icl_head tabpdl \
  --checkpoint_dir "$CKPT_DIR" \
  --save_temp_every 500 \
  --save_perm_every 2000 \
  --metrics_csv "$PROJECT_ROOT/training_metrics/mini_tabicl_stage1_finetune.csv"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] PDLC head-only fine-tuning failed with exit code $EXIT_CODE" >&2
else
  echo "[INFO] PDLC head-only fine-tuning finished successfully at $(date)"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
