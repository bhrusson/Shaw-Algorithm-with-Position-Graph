#!/bin/bash
#SBATCH --job-name=phir-QuantumVolume_20
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=phir-data/std_logs/QuantumVolume_20.log
#SBATCH --error=phir-data/err_logs/QuantumVolume_20.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_phir.py phir-data/input_qasms/QuantumVolume_20.qasm phir-data/output_qasms/QuantumVolume_20.qasm phir-data/output_pkls/QuantumVolume_20.pkl

