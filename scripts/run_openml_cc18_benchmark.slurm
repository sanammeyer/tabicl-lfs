#!/bin/bash

# SLURM job to run OpenML-CC18 10-fold CV benchmark
# Mirrors environment setup from run_tabicl_train.slurm

# SLURM Directives
# Adjust these to match your cluster/accounting needs
#SBATCH -J TabICL_CC18_Benchmark      # Job name
#SBATCH -o ./%x.%j.%N.out            # STDOUT log
#SBATCH -e ./%x.%j.%N.err            # STDERR log
#SBATCH --get-user-env               # Load user env
#SBATCH --export=NONE                # Don't inherit submit shell env
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std         # Partition/queue
#SBATCH --gpus=1                     # Number of GPUs per node
#SBATCH --ntasks=1                   # One task
#SBATCH --cpus-per-task=16           # CPU cores
#SBATCH --mem=64G                   # RAM
#SBATCH --time=120:00:00             # Max runtime
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT  # Email on key events (override with sbatch --mail-type)
#SBATCH --mail-user=sanamjeet.meyer@campus.lmu.de     # Destination email (override with sbatch --mail-user)

set -euo pipefail

# Load SLURM setup module (if required on your cluster)
module load slurm_setup || true

# --- Paths (overridable via env with sbatch --export=ALL,VAR=VAL,...) ---
PROJECT_ROOT="${PROJECT_ROOT:-/dss/dsshome1/0E/ra63pux2/Documents/thesis/code/tabicl-lfs}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.tabicl/bin/activate}"   # path to your venv activate script
PY_SRC_DIR="${PY_SRC_DIR:-$PROJECT_ROOT/src}"                          # so that `import tabicl` works
#CKPT_DIR_SA="${CKPT_DIR_SA:-$PROJECT_ROOT/checkpoints_mini_tabicl_stage2_sa}"                     # where checkpoints live
#CKPT_DIR_EA="${CKPT_DIR_EA:-$PROJECT_ROOT/checkpoints_mini_tabicl_stage2_ea}"                     # where checkpoints live
CKPT_DIR_PDLC="${CKPT_DIR_PDLC:-$PROJECT_ROOT/checkpoints_mini_tabicl_stage2_pdl}"                     # where checkpoints live
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

# Threading and NCCL defaults (safe for single-node jobs)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Resolve checkpoints (absolute paths)
CKPT_PDLC="$CKPT_DIR_PDLC/step-1000.ckpt"
#CKPT_SA="$CKPT_DIR_SA/step-1000.ckpt"


echo "[INFO] Checkpoint dir: $CKPT_DIR_PDLC"
printf "[INFO] Looking for checkpoint files:\n  - %s\n" "$CKPT_PDLC"

# Helper: run a single benchmark config
run_benchmark() {
  local model="$1"; shift
  echo "[RUN] openml_cc18_benchmark.py $model $*"
  srun -u python -u "$PROJECT_ROOT/openml_cc18_benchmark.py" --model "$model" "$@"
}

# Note about n_rows=None in user request:
# The benchmark script expects integers. To emulate "no limit", pass a very large value.
NO_LIMIT_ROWS=1000000000

FAIL=0

echo "[INFO] Starting OpenML-CC18 benchmark runs"

if [[ -f "$CKPT_PDLC" ]]; then
  run_benchmark tabicl \
    --checkpoint "$CKPT_PDLC" \
    --device cuda \
    --n_rows "$NO_LIMIT_ROWS" \
    --max_features 500 \
    --csv_postfix "pdlc_head" \
    --seed 42 || FAIL=1
else
  echo "[WARN] Missing checkpoint: $CKPT_PDLC — skipping this run"
fi

# Be tidy
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi

exit $FAIL

