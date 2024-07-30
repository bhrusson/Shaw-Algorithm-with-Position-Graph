#!/bin/bash
#SBATCH --job-name=phir-hubbard_3x2_0
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/hubbard_3x2_0.log
#SBATCH --error=phir-data/err_logs/hubbard_3x2_0.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/hubbard_3x2_0.qasm phir-data/output_qasms/hubbard_3x2_0.qasm phir-data/output_pkls/hubbard_3x2_0.pkl

