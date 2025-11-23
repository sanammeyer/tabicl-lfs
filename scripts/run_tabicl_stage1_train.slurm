#!/bin/bash

# SLURM job: Stage-1 end-to-end TabICL training (from pre-generated priors ONLY)
# - Loads priors from a fixed directory; aborts if unavailable
# - Enables elliptical attention in TFrow and TFicl
# - Uses 4x A100 GPUs on a single node via torchrun
# - Saves checkpoints every 5000 steps, keeping only the latest 3

# SLURM Directives
# Adjust cluster/partition to your environment if needed
#SBATCH -J TabICL_S1_E2E           # Job name
#SBATCH -o ./%x.%j.%N.out          # STDOUT log
#SBATCH -e ./%x.%j.%N.err          # STDERR log
#SBATCH --get-user-env             # Load user env
#SBATCH --export=NONE              # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std       # Partition/queue
#SBATCH --nodes=1                  # Single node
#SBATCH --gpus-per-node=4          # GPUs per node (A100 80GB x4)
#SBATCH --ntasks=1                 # One task
#SBATCH --cpus-per-task=32         # CPU cores
#SBATCH --mem=256G                 # RAM
#SBATCH --time=120:00:00           # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths (hard-coded) ---
PROJECT_ROOT="/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs"
VENV_ACTIVATE="$PROJECT_ROOT/.tabicl/bin/activate"
PY_SRC_DIR="$PROJECT_ROOT/src"
# Checkpoints saved here
CKPT_DIR="$PROJECT_ROOT/checkpoints_stage1"
# REQUIRED: Pre-generated priors directory (must exist and be readable)
PRIOR_DIR="/dss/lxclscratch/0E/ra63pux2/ra63pux2/tabicl_s1_priors"

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

# --- Validate prior directory strictly (abort if not usable) ---
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
echo "[INFO] Using pre-generated priors from: $PRIOR_DIR (files: $BATCH_COUNT)"

echo "[INFO] Launching Stage-1 end-to-end training (elliptical in TFrow+TFicl, from disk)"

# Checkpointing policy:
#  - Save every 5000 steps via --save_temp_every 5000
#  - Disable permanent checkpoints by setting --save_perm_every >> max_steps
#  - Keep only the latest 3 temporary checkpoints via --max_checkpoints 3

srun torchrun --standalone --nproc_per_node=4 \
  -m tabicl.train.run \
  --wandb_log False \
  --wandb_project TabICL \
  --wandb_name Stage1-E2E \
  --wandb_dir "$PROJECT_ROOT/wandb" \
  --wandb_mode disabled \
  --device cuda \
  --dtype float32 \
  --amp True \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 100000 \
  --batch_size 512 \
  --micro_batch_size 4 \
  --lr 1e-4 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.02 \
  --gradient_clipping 1.0 \
  --prior_dir "$PRIOR_DIR" \
  --load_prior_start 0 \
  --delete_after_load False \
  --prior_device cpu \
  --batch_size_per_gp 4 \
  --min_features 2 \
  --max_features 100 \
  --max_classes 10 \
  --max_seq_len 1024 \
  --min_train_size 0.1 \
  --max_train_size 0.9 \
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
  --freeze_col False \
  --freeze_row False \
  --freeze_icl False \
  --row_elliptical True \
  --icl_elliptical True \
  --checkpoint_dir "$CKPT_DIR" \
  --save_temp_every 5000 \
  --save_perm_every 1000000000 \
  --max_checkpoints 3

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE

