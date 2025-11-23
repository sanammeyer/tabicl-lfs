#!/bin/bash

# SLURM job array: Generate Stage-1 priors for indices 100000..159999 (60k batches)
# - Exactly 4 tasks, each creates 15,000 batches
# - Seeds are NOT offset per task (NP_SEED=42, TORCH_SEED=42 for all)
# - Constrained to 4 specified nodes; one task per node via --exclusive

# SLURM Directives
#SBATCH -J TabICL_S1_PriorsTail       # Job name
#SBATCH -o ./%x.%A_%a.%N.out          # STDOUT log
#SBATCH -e ./%x.%A_%a.%N.err          # STDERR log
#SBATCH --get-user-env                 # Load user env
#SBATCH --export=NONE                  # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std           # Partition/queue
#SBATCH --nodes=1                      # One node per task
#SBATCH --ntasks=1                     # One task per array index
#SBATCH --exclusive                    # Take whole node (ensures 1 task/node)
# If you must target specific nodes, pass --nodelist at submit time.
# Default: let Slurm place one exclusive task per available node.
#SBATCH --cpus-per-task=80             # Use all 80 CPU cores
#SBATCH --mem=128G                     # RAM per task
#SBATCH --time=144:00:00                # Max runtime per task
#SBATCH --array=0-3%4                  # 4 tasks total; run up to 4 concurrently
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de
set -euo pipefail

module load slurm_setup || true

# --- Paths ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"
PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-/dss/lxclscratch/0E/ra63pux2/ra63pux2/tabicl_s1_priors}"

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

mkdir -p "$PRIOR_SAVE_DIR" || true

# --------------------
# Fixed tail generation
# --------------------
BASE_OFFSET="${BASE_OFFSET:-100000}"     # first new index
CHUNK_SIZE="${CHUNK_SIZE:-15000}"       # batches per task (4*15000 = 60000)

IDX=${SLURM_ARRAY_TASK_ID:-0}
RESUME_FROM=$(( BASE_OFFSET + IDX * CHUNK_SIZE ))
NUM_BATCHES="$CHUNK_SIZE"

echo "[INFO] Array index: $IDX  Resume from: $RESUME_FROM  Num batches: $NUM_BATCHES"
echo "[INFO] Saving to: $PRIOR_SAVE_DIR"

# Stage-1 prior generation hyperparameters (mirror train_stage1.sh)
NP_SEED="${NP_SEED:-42}"              # SAME seed for all tasks (as requested)
TORCH_SEED="${TORCH_SEED:-42}"        # SAME seed for all tasks
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
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:--1}}"   # use all CPUs
THREADS_PER_GEN="${THREADS_PER_GEN:-1}"
DEVICE="${DEVICE:-cpu}"

ARGS=(
  --save_dir "$PRIOR_SAVE_DIR"
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

python -m tabicl.prior.genload "${ARGS[@]}"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Prior generation failed with exit code $EXIT_CODE" >&2
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
exit $EXIT_CODE
