#!/bin/bash
#SBATCH --job-name=TFXY_n16_s100
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/TFXY_n16_s100.log
#SBATCH --error=data/err_logs/TFXY_n16_s100.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/TFXY_n16_s100.qasm data/output_qasms/TFXY_n16_s100.qasm data/output_pkls/TFXY_n16_s100.pkl

