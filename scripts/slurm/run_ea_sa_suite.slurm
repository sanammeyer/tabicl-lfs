#!/bin/bash

# SLURM job: Run EA/SA comparison suite (behaviour, diagnostics, collapse)
# - Uses the same dataset list and checkpoints across all three scripts
# - Writes results to the scripts' built-in CSV outputs in results/

# SLURM Directives
#SBATCH -J EA_SA_Suite               # Job name
#SBATCH -o ./%x.%j.%N.out            # STDOUT log
#SBATCH -e ./%x.%j.%N.err            # STDERR log
#SBATCH --get-user-env               # Load user env
#SBATCH --export=NONE                # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std         # Partition/queue
#SBATCH --gpus=1                     # One GPU is sufficient for inference
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=16           # CPU threads
#SBATCH --mem=64G                    # RAM
#SBATCH --time=48:00:00              # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths (override via sbatch --export) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"

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

# Optional: OpenML cache
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$PROJECT_ROOT/.openml_cache}"
mkdir -p "$OPENML_CACHE_DIR" || true

# Mitigate CUDA fragmentation on large attention ops
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Shared config ---
DATASETS="${DATASETS:-23,48,750,906,24,40994,1063,1053,1485,1068,40668,11,1487,37,1475}"
SA_CKPT="${SA_CKPT:-$PROJECT_ROOT/checkpoints/mini_tabicl_stage2_sa/step-1000.ckpt}"
EA_CKPT="${EA_CKPT:-$PROJECT_ROOT/checkpoints/mini_tabicl_stage2_ea/step-1000_ea.ckpt}"
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-32}"
SEED="${SEED:-42}"

if [[ ! -f "$SA_CKPT" ]]; then
  echo "[ERROR] SA checkpoint not found: $SA_CKPT" >&2; exit 1; fi
if [[ ! -f "$EA_CKPT" ]]; then
  echo "[ERROR] EA checkpoint not found: $EA_CKPT" >&2; exit 1; fi

echo "[INFO] Running EA/SA suite on datasets: $DATASETS"
echo "[INFO] SA ckpt: $SA_CKPT"
echo "[INFO] EA ckpt: $EA_CKPT"
echo "[INFO] n_estimators: $N_ESTIMATORS"

# 1) Behaviour + geometry + robustness comparison
python scripts/analysis/compare_tabicl_sa_ea.py \
  --sa_checkpoint "$SA_CKPT" \
  --ea_checkpoint "$EA_CKPT" \
  --datasets "$DATASETS" \
  --n_estimators "$N_ESTIMATORS" \
  --device "$DEVICE" \
  --seed "$SEED"

# 2) EA/SA diagnostics (attention geometry + per-example diff)
python scripts/diagnostics/ea_sa_diagnostics.py \
  --datasets "$DATASETS" \
  --sa_checkpoint "$SA_CKPT" \
  --ea_checkpoint "$EA_CKPT" \
  --n_estimators "$N_ESTIMATORS" \
  --device "$DEVICE" \
  --seed "$SEED"

# 3) EA/SA collapse analysis
python scripts/analysis/ea_sa_collapse.py \
  --datasets "$DATASETS" \
  --sa_checkpoint "$SA_CKPT" \
  --ea_checkpoint "$EA_CKPT" \
  --n_estimators "$N_ESTIMATORS" \
  --device "$DEVICE" \
  --seed "$SEED"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] EA/SA suite failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
