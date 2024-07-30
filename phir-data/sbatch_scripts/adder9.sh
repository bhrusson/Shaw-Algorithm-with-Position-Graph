#!/bin/bash
#SBATCH --job-name=phir-adder9
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/adder9.log
#SBATCH --error=phir-data/err_logs/adder9.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/adder9.qasm phir-data/output_qasms/adder9.qasm phir-data/output_pkls/adder9.pkl

