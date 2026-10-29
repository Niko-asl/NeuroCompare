#!/bin/bash
#SBATCH --job-name=nc_preproc
#SBATCH --account RITTMAN-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/nc_preproc_%A_%a.out
#SBATCH --error=logs/nc_preproc_%A_%a.err
#
# NeuroCompare preprocessing (HD-BET skull strip -> FLIRT 12-dof to MNI152 1mm ->
# 2nd-98th percentile intensity normalisation) over a manifest, sharded across the array.
#
# USAGE (submit from the repo root, or from scripts/; slurm logs go to logs/ there):
#   mkdir -p logs
#   MANIFEST=.../fm_first_scan.csv OUTPUT_DIR=.../nc_prep_first sbatch slurm/preprocess_array.sh
# Set REPO=... to submit from anywhere else.
#
# Each task processes every N-th manifest row (N = array size) and writes
#   OUTPUT_DIR/final_processed/<cohort>/<cohort>_<sub>_<label>_<scan>_processed.nii.gz
#   OUTPUT_DIR/preprocessing_results_shardXXX.csv
# Finished scans are skipped, so failed or timed-out tasks can simply be resubmitted,
# e.g. sbatch --array=7,12 slurm/preprocess_array.sh. Keep the array size the same
# between submissions, or shards will be assigned different rows.
# After all tasks finish, build the splits:
#   python scripts/make_splits.py --scans 'OUTPUT_DIR/preprocessing_results_shard*.csv' ...

set -uo pipefail

module purge
module load rhel8/default-icl
module load miniconda3
module load fsl/6.0.7

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-frame}"

# ------------------------------------------------------------- configuration
# Repo checkout: REPO if set, else found from the submit directory (repo root or scripts/)
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -z "${REPO:-}" ]]; then
    if   [[ -r "$SUBMIT_DIR/scripts/preprocess.py" ]]; then REPO="$SUBMIT_DIR"
    elif [[ -r "$SUBMIT_DIR/preprocess.py" ]];         then REPO="$(dirname "$SUBMIT_DIR")"
    else REPO=""; fi
fi
MANIFEST="${MANIFEST:-/rds-d6/project/rds-nCu8GHKxWRw/niko_analysis/nc_manifests/fm_first_scan.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/rds-d6/project/rds-nCu8GHKxWRw/niko_analysis/nc_prep_first}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
RUN_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SHARD}"
WORK_DIR="${OUTPUT_DIR}/_logs/shard_$(printf '%03d' "$SHARD")"

export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Intermediates (HD-BET, FLIRT outputs) go to node-local scratch and are deleted per scan
export TMPDIR="${TMPDIR:-/tmp}"

# ----------------------------------------------------------------- preflight
[[ -r "$MANIFEST" ]] || { echo "FATAL: cannot read manifest $MANIFEST" >&2; exit 1; }
[[ -r "$REPO/scripts/preprocess.py" ]] || { echo "FATAL: no scripts/preprocess.py under REPO=$REPO" >&2; exit 1; }
command -v hd-bet >/dev/null || { echo "FATAL: hd-bet not on PATH in env '${CONDA_ENV:-frame}'" >&2; exit 1; }
command -v flirt  >/dev/null || { echo "FATAL: flirt not found (fsl module)" >&2; exit 1; }
[[ -r "${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz" ]] || {
    echo "FATAL: MNI template missing under FSLDIR='${FSLDIR:-unset}'" >&2; exit 1; }

mkdir -p "$WORK_DIR" || { echo "FATAL: cannot create $WORK_DIR" >&2; exit 1; }
probe="${OUTPUT_DIR}/.writetest_${RUN_TAG}"
touch "$probe" 2>/dev/null || { echo "FATAL: $OUTPUT_DIR is not writable" >&2; exit 1; }
rm -f "$probe"

# preprocess.py writes its log to the current directory; one directory per shard
# keeps the logs from interleaving.
cd "$WORK_DIR" || exit 1

cat <<INFO
manifest   : $MANIFEST
output dir : $OUTPUT_DIR
repo       : $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))
shard      : $SHARD of $NSHARDS
hd-bet     : $(command -v hd-bet)
threads    : $OMP_NUM_THREADS

INFO

python "$REPO/scripts/preprocess.py" \
    --manifest "$MANIFEST" \
    --path_col path --subject_col sub_id --label_col label --cohort_col Dataset --scan_col scan_id \
    --output_dir "$OUTPUT_DIR" \
    --shard_index "$SHARD" --num_shards "$NSHARDS"
status=$?

results="${OUTPUT_DIR}/$( (( NSHARDS > 1 )) && printf 'preprocessing_results_shard%03d.csv' "$SHARD" || echo preprocessing_results.csv)"
if [[ -r "$results" ]]; then
    python - "$results" <<'PY'
import sys, pandas as pd
r = pd.read_csv(sys.argv[1])
ok = r['final_file'].notna() if 'final_file' in r else pd.Series(False, index=r.index)
print(f"\nshard summary: {ok.sum()} done, {(~ok).sum()} failed of {len(r)}")
if (~ok).any() and 'error' in r:
    print(r.loc[~ok, 'error'].value_counts().to_string())
PY
fi
exit $status
