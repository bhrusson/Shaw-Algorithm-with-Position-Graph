#!/bin/bash
#SBATCH --job-name=pam-phir-hubbard_3x2_0
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=pam-phir-data/std_logs/hubbard_3x2_0.log
#SBATCH --error=pam-phir-data/err_logs/hubbard_3x2_0.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_pam_phir.py pam-phir-data/input_qasms/hubbard_3x2_0.qasm pam-phir-data/output_qasms/hubbard_3x2_0.qasm pam-phir-data/output_pkls/hubbard_3x2_0.pkl

