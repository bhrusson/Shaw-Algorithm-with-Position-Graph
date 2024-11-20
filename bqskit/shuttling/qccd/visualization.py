import pickle
import matplotlib.pyplot as plt

circuit_lst = [
    "QAOA_16_compiled",
    "QuantumVolume_16",
    "QFT_16_compiled",
    "TFIM_n16_s100_compiled",
    "TFXY_n16_s100_compiled",
    "QAOA_20_compiled",
    "QuantumVolume_20",
    "QFT_20_compiled"
]

architecture_lst = [
    "H",
    "G2x3"
]

parameter_set = [
    ["4", "5"],
    ["5", "6"]
]

for circuit_idx in range(len(circuit_lst)):
    if circuit_idx < 5:
        with open(f"{circuit_lst[circuit_idx]}_{architecture_lst}", "rb") as input_file:
            data = pickle.load(input_file)

result = [
          runtime, compile_time,
          data["instruction_list"],
          output_circuit,
          output_circuit.gate_counts,
          data['initial_ion_assignment_qccd'],
          data['initial_mapping'],
          data['final_mapping'],
          machine_model
          ]