#!/bin/bash

# SLURM job: Generate Stage-1 prior datasets to disk
# - Mirrors the "Saving to disk" block in scripts/sh/train_stage1.sh
# - CPU-only generation; uses SLURM_CPUS_PER_TASK for parallelism

# SLURM Directives
#SBATCH -J mini-TabICL_S2_Priors          # Job name
#SBATCH -o ./%x.%j.%N.out            # STDOUT log
#SBATCH -e ./%x.%j.%N.err            # STDERR log
#SBATCH --get-user-env               # Load user env
#SBATCH --export=NONE                # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std         # Partition/queue
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # One task
#SBATCH --cpus-per-task=36           # CPU cores used for generation
#SBATCH --mem=128G                   # RAM (adjust as needed)
#SBATCH --time=100:00:00              # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths (override via sbatch --export) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"
PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-/dss/lxclscratch/0E/ra63pux2/ra63pux2/mini_tabicl_s2_priors}"

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

# Choose a writable save directory from candidates: PRIOR_SAVE_DIR, SCRATCH, SCRATCH_DSS, TMPDIR, TMPDIR_LOCAL
SAVE_DIR="$PRIOR_SAVE_DIR"
for cand in "$PRIOR_SAVE_DIR" "${SCRATCH:-}" "${SCRATCH_DSS:-}" "${TMPDIR:-}" "${TMPDIR_LOCAL:-}"; do
  if [[ -n "$cand" ]] && mkdir -p "$cand" 2>/dev/null; then
    SAVE_DIR="$cand"
    break
  fi
done
if [[ ! -d "$SAVE_DIR" ]]; then
  echo "[ERROR] No writable save directory found. Set PRIOR_SAVE_DIR to a path you own." >&2
  exit 1
fi

# --- Stage-2 prior generation hyperparameters (for mini tabicl) ---
NP_SEED="${NP_SEED:-42}"
TORCH_SEED="${TORCH_SEED:-42}"
NUM_BATCHES="${NUM_BATCHES:-5000}"
RESUME_FROM="${RESUME_FROM:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-1}"
MIN_FEATURES="${MIN_FEATURES:-2}"
MAX_FEATURES="${MAX_FEATURES:-100}"
MAX_CLASSES="${MAX_CLASSES:-10}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:- 1000}"        # leave empty to use default (None)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-10000}"
MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-0.5}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-0.9}"
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:--1}}"
THREADS_PER_GEN="${THREADS_PER_GEN:-1}"
SEQ_LEN_PER_GP="${SEQ_LEN_PER_GP:-True}"
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
  --seq_len_per_gp "$SEQ_LEN_PER_GP"
  --device "$DEVICE"
)

# Conditionally add --min_seq_len if provided (Stage-1 leaves it None)
if [[ -n "$MIN_SEQ_LEN" ]]; then
  ARGS+=(--min_seq_len "$MIN_SEQ_LEN")
fi

echo "[INFO] Generating Stage-1 priors to: $SAVE_DIR"
python -m tabicl.prior.genload "${ARGS[@]}"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Prior generation failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
