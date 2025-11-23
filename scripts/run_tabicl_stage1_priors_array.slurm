#!/bin/bash

# SLURM job array: parallel Stage-1 prior generation across many nodes
# - Splits total batches into disjoint contiguous chunks per array task
# - Safe to write into one shared PRIOR_SAVE_DIR as long as index ranges don't overlap
# - Tune array size and %concurrency to match cluster capacity and filesystem load

# SLURM Directives
#SBATCH -J TabICL_S1_PriorsArr        # Job name
#SBATCH -o ./%x.%A_%a.%N.out          # STDOUT log (A=array jobid, a=array index)
#SBATCH -e ./%x.%A_%a.%N.err          # STDERR log
#SBATCH --get-user-env                 # Load user env
#SBATCH --export=NONE                  # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std           # CPU-capable partition
#SBATCH --nodes=1                      # One node per array task
#SBATCH --ntasks=1                     # One task per array index
#SBATCH --cpus-per-task=32             # CPU cores for joblib
#SBATCH --mem=128G                     # RAM (adjust if needed)
#SBATCH --time=48:00:00                # Max runtime per array task
#SBATCH --array=0-19%8                 # 20 chunks total, 8 concurrent (override at submit time)
#SBATCH --mail-type=FAIL,TIME_LIMIT              
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths (override via sbatch --export) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"
PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-$PROJECT_ROOT/data/stage1_priors}"

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

# Resolve save dir with fallback to node-local scratch (TMPDIR) if not writable
SAVE_DIR="$PRIOR_SAVE_DIR"
if ! mkdir -p "$SAVE_DIR" 2>/dev/null; then
  if [[ -n "${TMPDIR:-}" ]]; then
    SAVE_DIR="${TMPDIR%/}/tabicl_s1_priors"
    mkdir -p "$SAVE_DIR"
    echo "[WARN] PRIOR_SAVE_DIR not writable; using TMPDIR: $SAVE_DIR"
    # Optionally sync back to persistent storage after generation
    USE_SYNC_OUT=1
    SYNC_OUT_DIR="${SYNC_OUT_DIR:-$PRIOR_SAVE_DIR}"
  else
    echo "[ERROR] Neither PRIOR_SAVE_DIR ($PRIOR_SAVE_DIR) nor TMPDIR are writable. Set PRIOR_SAVE_DIR to a path you own." >&2
    exit 1
  fi
fi

# --------------------
# Chunking parameters
# --------------------
# Total number of batches to generate across the entire array (override via env)
TOTAL_BATCHES="${TOTAL_BATCHES:-100000}"
# Number of batches per array task (chunk size)
CHUNK_SIZE="${CHUNK_SIZE:-5000}"

IDX=${SLURM_ARRAY_TASK_ID:-0}
RESUME_FROM=$(( IDX * CHUNK_SIZE ))

# Compute this task's NUM_BATCHES bounded by TOTAL_BATCHES
if [[ "$TOTAL_BATCHES" -gt 0 ]]; then
  if [[ $RESUME_FROM -ge "$TOTAL_BATCHES" ]]; then
    echo "[INFO] Nothing to do for array index $IDX (start $RESUME_FROM >= total $TOTAL_BATCHES). Exiting."
    exit 0
  fi
  REMAIN=$(( TOTAL_BATCHES - RESUME_FROM ))
  if [[ $REMAIN -lt $CHUNK_SIZE ]]; then
    NUM_BATCHES=$REMAIN
  else
    NUM_BATCHES=$CHUNK_SIZE
  fi
else
  # Unlimited mode (not recommended); each task does CHUNK_SIZE
  NUM_BATCHES=$CHUNK_SIZE
fi

echo "[INFO] Array index: $IDX  Resume from: $RESUME_FROM  Num batches: $NUM_BATCHES"

# --- Stage-1 prior generation hyperparameters (mirror train_stage1.sh) ---
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-4}"
MIN_FEATURES="${MIN_FEATURES:-2}"
MAX_FEATURES="${MAX_FEATURES:-100}"
MAX_CLASSES="${MAX_CLASSES:-10}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-}"        # leave empty to use default (None)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-0.1}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-0.9}"
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:--1}}"
THREADS_PER_GEN="${THREADS_PER_GEN:-1}"
DEVICE="${DEVICE:-cpu}"

ARGS=(
  --save_dir "$SAVE_DIR"
  --np_seed "$NP_SEED"
  --torch_seed "$TORCH_SEED"
  --num_batches "$NUM_BATCHES"
  --resume_from "$RESUME_FROM"
  --batch_size "$BATCH_SIZE"
  --batch_size_per_gp "$BATCH_SIZE_PER_GP"
  --prior_type "$PRIOR_TYPE"
  --min_features "$MIN_FEATURES"
  --max_features "$MAX_FEATURES"
  --max_classes "$MAX_CLASSES"
  --max_seq_len "$MAX_SEQ_LEN"
  --min_train_size "$MIN_TRAIN_SIZE"
  --max_train_size "$MAX_TRAIN_SIZE"
  --n_jobs "$N_JOBS"
  --num_threads_per_generate "$THREADS_PER_GEN"
  --device "$DEVICE"
)

if [[ -n "$MIN_SEQ_LEN" ]]; then
  ARGS+=(--min_seq_len "$MIN_SEQ_LEN")
fi

echo "[INFO] Generating Stage-1 priors to: $SAVE_DIR"
python -m tabicl.prior.genload "${ARGS[@]}"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Prior generation failed with exit code $EXIT_CODE" >&2
else
  # If we used TMPDIR fallback, sync results to persistent target (best-effort, ignore conflicts)
  if [[ "${USE_SYNC_OUT:-0}" -eq 1 && -n "${SYNC_OUT_DIR:-}" ]]; then
    echo "[INFO] Syncing generated batches to: $SYNC_OUT_DIR"
    mkdir -p "$SYNC_OUT_DIR" || true
    rsync -a --ignore-existing "$SAVE_DIR/" "$SYNC_OUT_DIR/" || true
  fi
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
