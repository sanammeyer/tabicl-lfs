#!/bin/bash

# SLURM job to train TabICL (Stage-1/2) with Stage-3 frozen
# Mirrors scripts/run_tabicl_train.slurm but flips freeze flags and enables the probe.

# SLURM Directives
#SBATCH -J TabICL_S1S2_Train         # Job name
#SBATCH -o ./%x.%j.%N.out           # STDOUT log
#SBATCH -e ./%x.%j.%N.err           # STDERR log
#SBATCH --get-user-env              # Load user env
#SBATCH --export=NONE               # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std        # Partition/queue
#SBATCH --gpus=4                    # Number of GPUs per node
#SBATCH --ntasks=1                  # One task
#SBATCH --cpus-per-task=32          # CPU cores to feed the GPU
#SBATCH --mem=256G                  # RAM
#SBATCH --time=24:00:00             # Short run for debugging stage-1/2
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT  # Email on key events (override with sbatch --mail-type)
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de     # Destination email (override with sbatch --mail-user)
set -euo pipefail

module load slurm_setup || true

# --- Paths (override via sbatch --export) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DEFAULT_STAGE3_CKPT="$CKPT_DIR/tabicl-classifier-v1.1-0506.ckpt"
TRAIN_MODULE="tabicl.train.run"

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

# Selection policy:
# 1) If CHECKPOINT_PATH is set, use it.
# 2) Else if default Stage-3 ckpt exists, use it.
# 3) Else: if exactly one ckpt exists, use it; otherwise start from scratch (trainer will auto-pick latest step-* if any).
if [[ -z "$CHECKPOINT_PATH" && -d "$CKPT_DIR" ]]; then
  if [[ -f "$DEFAULT_STAGE3_CKPT" ]]; then
    CHECKPOINT_PATH="$DEFAULT_STAGE3_CKPT"
    echo "[INFO] Using default Stage-3 checkpoint: $CHECKPOINT_PATH"
  else
    mapfile -t __ckpts < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "*.ckpt" | sort)
    if [[ ${#__ckpts[@]} -eq 1 ]]; then
      CHECKPOINT_PATH="${__ckpts[0]}"
      echo "[INFO] Auto-detected single checkpoint: $CHECKPOINT_PATH"
    elif [[ ${#__ckpts[@]} -eq 0 ]]; then
      echo "[INFO] No .ckpt files found in $CKPT_DIR; starting from scratch." >&2
    else
      echo "[WARN] Multiple .ckpt files found; not passing --checkpoint_path explicitly." >&2
    fi
  fi
fi

CHECKPOINT_FLAG=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_FLAG=(--checkpoint_path "$CHECKPOINT_PATH")
fi

echo "[INFO] Launching Stage-1/2 training (Stage-3 frozen)"
srun torchrun --standalone --nproc_per_node=4 \
  -m "$TRAIN_MODULE" \
  --device cuda \
  --dtype bfloat16 \
  --amp True \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 500 \
  --batch_size 256 \
  --micro_batch_size 1 \
  --lr 8e-5 \
  --scheduler polynomial_decay_warmup \
  --warmup_proportion 0.05 \
  --poly_decay_lr_end 5e-6 \
  --poly_decay_power 2.0 \
  --gradient_clipping 1.0 \
  --prior_type mix_scm \
  --prior_device cpu \
  --batch_size_per_gp 2 \
  --min_features 2 \
  --max_features 50 \
  --max_classes 10 \
  --min_seq_len 1000 \
  --max_seq_len 5000 \
  --log_seq_len True \
  --seq_len_per_gp False \
  --min_train_size 0.5 \
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
  --checkpoint_dir "$CKPT_DIR" \
  "${CHECKPOINT_FLAG[@]}" \
  --only_load_model True \
  --freeze_col False \
  --freeze_row False \
  --freeze_icl True \
  --row_elliptical True \
  --icl_elliptical False \
  --icl_elliptical_lr_mult 3.0 \
  --icl_elliptical_weight_decay 0.0 \
  --wandb_log False \
  --wandb_mode disabled \
  --metrics_csv "$PROJECT_ROOT/results/s1s2_train_metrics.csv" \
  --probe_every 50 \
  --probe_batch_size 10 \
  --probe_seed 1337

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Training failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
