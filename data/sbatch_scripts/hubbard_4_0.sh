#!/bin/bash
#SBATCH --job-name=hubbard_4_0
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/hubbard_4_0.log
#SBATCH --error=data/err_logs/hubbard_4_0.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/hubbard_4_0.qasm data/output_qasms/hubbard_4_0.qasm data/output_pkls/hubbard_4_0.pkl

