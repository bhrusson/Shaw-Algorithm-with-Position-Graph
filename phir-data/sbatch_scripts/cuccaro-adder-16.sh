#!/bin/bash
#SBATCH --job-name=phir-cuccaro-adder-16
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/cuccaro-adder-16.log
#SBATCH --error=phir-data/err_logs/cuccaro-adder-16.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/cuccaro-adder-16.qasm phir-data/output_qasms/cuccaro-adder-16.qasm phir-data/output_pkls/cuccaro-adder-16.pkl

