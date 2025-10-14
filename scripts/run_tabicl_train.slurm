#!/bin/bash

# SLURM job to train TabICL with torchrun
# Adjust paths below to match your environment.

# SLURM Directives
#SBATCH -J TabICL_Train             # Job name
#SBATCH -o ./%x.%j.%N.out           # STDOUT log
#SBATCH -e ./%x.%j.%N.err           # STDERR log
#SBATCH --get-user-env              # Load user env
#SBATCH --export=NONE               # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std        # Partition/queue
#SBATCH --gpus=4                    # Number of GPUs per node
#SBATCH --ntasks=1                  # One task
#SBATCH --cpus-per-task=32           # CPU cores to feed the GPU
#SBATCH --mem=256G                   # RAM
#SBATCH --time=120:00:00             # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT  # Email on key events (override with sbatch --mail-type)
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de     # Destination email (override with sbatch --mail-user)

set -euo pipefail

# Load SLURM setup module (if required on your cluster)
module load slurm_setup || true

# --- User-configurable paths (overridable via env with sbatch --export) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"   # path to your venv activate script
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"                          # so that `import tabicl` works

# Checkpointing (adjust or override via env)
# Use a single repo-local ./checkpoints directory for both loading and saving
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints}"
# Optionally override exact checkpoint file (e.g., a .ckpt file).
# Leave empty by default; we'll auto-detect a single .ckpt when not provided.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# --- Derived and constant config ---
TRAIN_MODULE="tabicl.train.run"   # module passed to torchrun with -m

echo "[INFO] Working directory before cd: $(pwd)"
cd "$PROJECT_ROOT"
echo "[INFO] Project root: $(pwd)"

echo "[INFO] Looking for virtual environment at: $VENV_ACTIVATE"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  echo "[INFO] Virtual environment activated: ${VIRTUAL_ENV:-unknown}"
else
  echo "[ERROR] Virtual environment not found at $VENV_ACTIVATE" >&2
  echo "[HINT] Create it and install deps, or point VENV_ACTIVATE to the right path." >&2
  exit 1
fi

echo "[INFO] Python: $(which python3)"
echo "[INFO] Torch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "not found")"

# Ensure Python finds the src/ package path
export PYTHONPATH="$PY_SRC_DIR:${PYTHONPATH:-}"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# Threading and NCCL defaults for single-node stability
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Create checkpoint directory if it uses a local absolute path
if [[ "$CKPT_DIR" == /* ]]; then
  mkdir -p "$CKPT_DIR" || true
fi

# If there's exactly one checkpoint in CKPT_DIR and no explicit CHECKPOINT_PATH provided,
# auto-detect and use it. This matches setups where each stage keeps a single .ckpt file.
if [[ -z "$CHECKPOINT_PATH" && -d "$CKPT_DIR" ]]; then
  mapfile -t __ckpts < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "*.ckpt" | sort)
  if [[ ${#__ckpts[@]} -eq 1 ]]; then
    CHECKPOINT_PATH="${__ckpts[0]}"
    echo "[INFO] Auto-detected single checkpoint: $CHECKPOINT_PATH"
  elif [[ ${#__ckpts[@]} -eq 0 ]]; then
    echo "[INFO] No .ckpt files found in $CKPT_DIR; starting from scratch." >&2
  else
    echo "[WARN] Multiple .ckpt files found in $CKPT_DIR; will NOT pass --checkpoint_path so training auto-loads latest step-* from --checkpoint_dir" >&2
  fi
fi

echo "[INFO] Starting torchrun TabICL training"
echo "[INFO] Checkpoint dir: $CKPT_DIR"
if [[ -n "$CHECKPOINT_PATH" ]]; then
  echo "[INFO] Using checkpoint_path: $CHECKPOINT_PATH"
else
  echo "[INFO] No explicit checkpoint_path; training will auto-detect latest in $CKPT_DIR"
fi

# Build optional checkpoint arg if a concrete file path is provided
CHECKPOINT_FLAG=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_FLAG=(--checkpoint_path "$CHECKPOINT_PATH")
fi

# Run training (stage-2 style per user command)
srun torchrun --standalone --nproc_per_node=4 \
  -m "$TRAIN_MODULE" \
  --device cuda \
  --dtype bfloat16 \
  --amp True \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 2000 \
  --batch_size 512 \
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
  --max_features 100 \
  --max_classes 10 \
  --min_seq_len 1000 \
  --max_seq_len 40000 \
  --log_seq_len True \
  --seq_len_per_gp True \
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
  --freeze_col True \
  --freeze_row True \
  --icl_elliptical True 

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[INFO] Training finished successfully at $(date)"
else
  echo "[ERROR] Training failed with exit code $EXIT_CODE at $(date)" >&2
fi

# If a venv is active, deactivate to be tidy
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi

exit $EXIT_CODE
