#!/bin/bash
#SBATCH --job-name=toffoli
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/toffoli.log
#SBATCH --error=data/err_logs/toffoli.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/toffoli.qasm data/output_qasms/toffoli.qasm data/output_pkls/toffoli.pkl

