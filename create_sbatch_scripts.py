script = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output={out_log}
#SBATCH --error={err_log}

module load python &> /dev/null
conda activate bqskit-shuttling &> /dev/null

python run_pam_phir.py {input_qasm} {output_qasm} {output_pkl}

"""

qasms = ["adder9_trapsize3.qasm", "cuccaro-adder-16.qasm", "fredkin.qasm", "toffoli.qasm", "Grover_5.qasm", "Grover_8.qasm", "heisenberg-16-20.qasm", "hubbard_2x2_0.qasm", "hubbard_3x2_0.qasm", "hubbard_4_0.qasm", "incrementer-16.qasm", "PhaseEstimator_5.qasm", "PhaseEstimator_8.qasm", "QAOA_16.qasm", "QAOA_20.qasm", "QFT_16.qasm", "QFT_20.qasm", "QuantumVolume_16.qasm", "QuantumVolume_20.qasm", "TFIM_n16_s100.qasm", "TFXY_n16_s100.qasm"]

for qasm in qasms:
    name = qasm.split(".")[0]
    composed = script.format(
        name="pam-phir-" + name,
        out_log="pam-phir-data/std_logs/" + name + ".log",
        err_log="pam-phir-data/err_logs/" + name + ".err",
        input_qasm="pam-phir-data/input_qasms/" + qasm,
        output_qasm="pam-phir-data/output_qasms/" + qasm,
        output_pkl="pam-phir-data/output_pkls/" + name + ".pkl",
    )

    with open("pam-phir-data/sbatch_scripts/" + name + ".sh", "w") as f:
        f.write(composed)

