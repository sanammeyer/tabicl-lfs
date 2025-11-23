#!/bin/bash

# SLURM job: EA vs SA diagnostics on selected OpenML datasets
# - Runs scripts/ea_sa_diagnostics.py over a curated list of datasets
# - Uses Stage-2 SA/EA checkpoints (override via env)

# SLURM Directives
#SBATCH -J EA_SA_Diagnostics         # Job name
#SBATCH -o ./%x.%j.%N.out            # STDOUT log
#SBATCH -e ./%x.%j.%N.err            # STDERR log
#SBATCH --get-user-env               # Load user env
#SBATCH --export=NONE                # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std         # Partition/queue
#SBATCH --gpus=1                     # One GPU is sufficient for inference
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=16           # CPU threads for data prep/IO
#SBATCH --mem=64G                    # RAM
#SBATCH --time=24:00:00              # Max runtime
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

# Optional: Use a writable cache for OpenML downloads
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$PROJECT_ROOT/.openml_cache}"
mkdir -p "$OPENML_CACHE_DIR" || true

# --- Checkpoints (override via env) ---
SA_CKPT="${SA_CKPT:-$PROJECT_ROOT/checkpoints_mini_tabicl_stage2_sa/step-1000.ckpt}"
EA_CKPT="${EA_CKPT:-$PROJECT_ROOT/checkpoints_mini_tabicl_stage2_ea/step-1000_ea.ckpt}"
if [[ ! -f "$SA_CKPT" ]]; then
  echo "[ERROR] SA checkpoint not found: $SA_CKPT" >&2; exit 1; fi
if [[ ! -f "$EA_CKPT" ]]; then
  echo "[ERROR] EA checkpoint not found: $EA_CKPT" >&2; exit 1; fi

# --- Dataset list ---
# Small/mixed/multiclass and numeric medium/large
# 23(cmc), 48(tae), 750(pm10), 906(chscase_census5), 24(mushroom)
# EA weaknesses (binary/classic, imbalance, high-dim)
# 40994(climate-model-simulation-crashes), 1063(kc2), 1053(jm1), 1485(madelon), 1068(pc1)
# EA strengths (structured/multiclass, classic small, imbalance cases)
# 40668(connect-4), 11(balance-scale), 1487(ozone-level-8hr), 37(diabetes), 1475(first-order-theorem-proving)
DATASETS="${DATASETS:-23,48,750,906,24,40994,1063,1053,1485,1068,40668,11,1487,37,1475}"

# --- Other toggles ---
DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-32}"
SEED="${SEED:-42}"
EA_STRENGTH="${EA_STRENGTH:-1.0}"

echo "[INFO] Running EA/SA diagnostics"
echo "[INFO] Datasets: $DATASETS"
echo "[INFO] SA ckpt: $SA_CKPT"
echo "[INFO] EA ckpt: $EA_CKPT"

srun python scripts/ea_sa_diagnostics.py \
  --datasets "$DATASETS" \
  --sa_checkpoint "$SA_CKPT" \
  --ea_checkpoint "$EA_CKPT" \
  --device "$DEVICE" \
  --n_estimators "$N_ESTIMATORS" \
  --seed "$SEED" \
  --ea_strength "$EA_STRENGTH"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Diagnostics failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
