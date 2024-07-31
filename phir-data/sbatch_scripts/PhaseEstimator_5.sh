#!/bin/bash
#SBATCH --job-name=phir-PhaseEstimator_5
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/PhaseEstimator_5.log
#SBATCH --error=phir-data/err_logs/PhaseEstimator_5.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/PhaseEstimator_5.qasm phir-data/output_qasms/PhaseEstimator_5.qasm phir-data/output_pkls/PhaseEstimator_5.pkl

