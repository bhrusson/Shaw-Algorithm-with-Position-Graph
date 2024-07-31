#!/bin/bash
#SBATCH --job-name=adder9
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/adder9.log
#SBATCH --error=data/err_logs/adder9.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/adder9.qasm data/output_qasms/adder9.qasm data/output_pkls/adder9.pkl

