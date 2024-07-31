#!/bin/bash
#SBATCH --job-name=PhaseEstimator_8
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/PhaseEstimator_8.log
#SBATCH --error=data/err_logs/PhaseEstimator_8.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/PhaseEstimator_8.qasm data/output_qasms/PhaseEstimator_8.qasm data/output_pkls/PhaseEstimator_8.pkl

