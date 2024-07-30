#!/bin/bash
#SBATCH --job-name=phir-toffoli
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/toffoli.log
#SBATCH --error=phir-data/err_logs/toffoli.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/toffoli.qasm phir-data/output_qasms/toffoli.qasm phir-data/output_pkls/toffoli.pkl

