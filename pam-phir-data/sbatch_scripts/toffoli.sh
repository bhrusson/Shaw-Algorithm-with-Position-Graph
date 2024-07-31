#!/bin/bash
#SBATCH --job-name=pam-phir-toffoli
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=pam-phir-data/std_logs/toffoli.log
#SBATCH --error=pam-phir-data/err_logs/toffoli.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_pam_phir.py pam-phir-data/input_qasms/toffoli.qasm pam-phir-data/output_qasms/toffoli.qasm pam-phir-data/output_pkls/toffoli.pkl

