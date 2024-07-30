#!/bin/bash
#SBATCH --job-name=QuantumVolume_16
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/QuantumVolume_16.log
#SBATCH --error=data/err_logs/QuantumVolume_16.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/QuantumVolume_16.qasm data/output_qasms/QuantumVolume_16.qasm data/output_pkls/QuantumVolume_16.pkl

