#!/bin/bash
#SBATCH --job-name=Grover_8
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/Grover_8.log
#SBATCH --error=data/err_logs/Grover_8.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/Grover_8.qasm data/output_qasms/Grover_8.qasm data/output_pkls/Grover_8.pkl

