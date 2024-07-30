#!/bin/bash
#SBATCH --job-name=phir-heisenberg-16-20
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/heisenberg-16-20.log
#SBATCH --error=phir-data/err_logs/heisenberg-16-20.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/heisenberg-16-20.qasm phir-data/output_qasms/heisenberg-16-20.qasm phir-data/output_pkls/heisenberg-16-20.pkl

