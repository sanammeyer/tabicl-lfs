#!/bin/bash
#
# Run OpenML-CC18 benchmark twice (TabPDL checkpoint):
#  1) Fixed PDLC top-k = 32
#  2) Per-dataset tuned PDLC top-k (k scaled with dataset size via best_frac)
#
# Usage:
#   sbatch scripts/slurm/run_openml_cc18_pdl_topk32_and_tuned.slurm
#
# Overridable env vars (sbatch --export=ALL,VAR=VAL,...):
#   PROJECT_ROOT, VENV_ACTIVATE, PY_SRC_DIR, CKPT_DIR, CKPT_STEP, OUT_DIR
#   N_ESTIMATORS, BATCH_SIZE, DEVICE, LIMIT, FOLDS, SEED
#   TOPK_FIXED, TOPK_TUNE_METRIC, TOPK_TUNE_VAL_FRAC, TOPK_TUNE_ABS, TOPK_TUNE_FRACS
#

#SBATCH -J TabPDL_CC18_TopK
#SBATCH -o ./%x.%j.%N.out
#SBATCH -e ./%x.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00

set -euo pipefail

module load slurm_setup || true

PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/mini_tabicl_stage2_pdl_posterior_avg}"
CKPT_STEP="${CKPT_STEP:-step-1000.ckpt}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results}"

DEVICE="${DEVICE:-cuda}"
N_ESTIMATORS="${N_ESTIMATORS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
FOLDS="${FOLDS:-10}"

TOPK_FIXED="${TOPK_FIXED:-32}"
TOPK_TUNE_METRIC="${TOPK_TUNE_METRIC:-f1_macro}"
TOPK_TUNE_VAL_FRAC="${TOPK_TUNE_VAL_FRAC:-0.2}"
TOPK_TUNE_ABS="${TOPK_TUNE_ABS:-8,16,32,64,128,256,512,1024,2048}"
TOPK_TUNE_FRACS="${TOPK_TUNE_FRACS:-0,0.001,0.002,0.005,0.01,0.02,0.05,0.1}"

RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"

echo "[INFO] Working directory before cd: $(pwd)"
cd "$PROJECT_ROOT"
echo "[INFO] Project root: $(pwd)"

echo "[INFO] Activating venv: $VENV_ACTIVATE"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "[ERROR] Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

export PYTHONPATH="$PY_SRC_DIR:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"

CKPT="$CKPT_DIR/$CKPT_STEP"
echo "[INFO] Checkpoint: $CKPT"
if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Missing checkpoint: $CKPT" >&2
  exit 1
fi

# OpenML cache: prefer local scratch if available (optional)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export OPENML_CACHE_DIRECTORY="${OPENML_CACHE_DIRECTORY:-$SLURM_TMPDIR/openml_cache}"
  mkdir -p "$OPENML_CACHE_DIRECTORY"
  echo "[INFO] OPENML_CACHE_DIRECTORY=$OPENML_CACHE_DIRECTORY"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NO_LIMIT_ROWS=1000000000

common_args=(
  --model tabicl
  --checkpoint "$CKPT"
  --device "$DEVICE"
  --n_estimators "$N_ESTIMATORS"
  --batch_size "$BATCH_SIZE"
  --n_rows "$NO_LIMIT_ROWS"
  --max_features 500
  --max_classes 10
  --seed "$SEED"
  --folds "$FOLDS"
  --pdlc_agg posterior_avg
)

if [[ -n "$LIMIT" ]]; then
  common_args+=( --limit "$LIMIT" )
fi

echo "[INFO] Run 1/2: fixed top-k=$TOPK_FIXED"
srun -u python -u scripts/benchmarks/openml_cc18_benchmark.py \
  "${common_args[@]}" \
  --pdlc_topk "$TOPK_FIXED" \
  --output "$OUT_DIR/cc18_pdl_topk${TOPK_FIXED}_${RUN_TAG}.csv"

echo "[INFO] Run 2/2: per-dataset tuned top-k (scaled with dataset size)"
srun -u python -u scripts/benchmarks/openml_cc18_benchmark.py \
  "${common_args[@]}" \
  --pdlc_topk_tune \
  --pdlc_topk_tune_metric "$TOPK_TUNE_METRIC" \
  --pdlc_topk_tune_val_frac "$TOPK_TUNE_VAL_FRAC" \
  --pdlc_topk_tune_abs "$TOPK_TUNE_ABS" \
  --pdlc_topk_tune_fracs "$TOPK_TUNE_FRACS" \
  --pdlc_topk_tune_cache "$OUT_DIR/cc18_pdl_topk_tuned_cache_${RUN_TAG}.json" \
  --output "$OUT_DIR/cc18_pdl_topk_tuned_${RUN_TAG}.csv"

echo "[INFO] Done."

