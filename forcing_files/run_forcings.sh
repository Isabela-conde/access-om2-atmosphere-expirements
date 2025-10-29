#!/bin/bash
#PBS -N jra555_saving
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l ncpus=28
#PBS -l mem=252GB
#PBS -l storage=gdata/e14+gdata/ua8+gdata/fs38+gdata/hh5+gdata/ia39+gdata/vk83+gdata/v45+gdata/ik11+gdata/qv56+gdata/rt52+gdata/xp65
#PBS -j oe

set -euo pipefail
echo "[$(date)] Job starting on $(hostname)"
echo "[$(date)] PBS job ID: $PBS_JOBID"
echo "[$(date)] Working dir: $PBS_O_WORKDIR"

module use /g/data/hh5/public/modules
module load conda/analysis3-24.07

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$PBS_O_WORKDIR"

PYFILE="save_trend_mean.py"

echo "[$(date)] Running: python -u ${PYFILE}"
python -u "${PYFILE}"

echo "[$(date)] Job finished."

