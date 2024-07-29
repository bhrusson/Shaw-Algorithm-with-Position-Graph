from bqskit.ext import H1_1Model
from bqskit import Circuit, compile

num_qudits = 20
circuit_type = "QuantumVolume"
cir = Circuit.from_file(f"experiments/results/experiment_circuits"
                        f"/input_circuits/{circuit_type}_{num_qudits}.qasm")
compiled_circuit = compile(cir, model=H1_1Model, optimization_level=3)

compiled_circuit.save(f"experiments/results/experiment_circuits/"
                      f"output_circuits/{circuit_type}_{num_qudits}_bqskit_compiled.qasm")