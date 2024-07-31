#!/bin/bash
#SBATCH --job-name=fredkin
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=data/std_logs/fredkin.log
#SBATCH --error=data/err_logs/fredkin.err

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run.py data/input_qasms/fredkin.qasm data/output_qasms/fredkin.qasm data/output_pkls/fredkin.pkl

