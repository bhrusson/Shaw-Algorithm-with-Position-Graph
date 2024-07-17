from bqskit import Circuit

circuit_name = "test"
print(f"Scheduling {circuit_name}.")
circuit = Circuit.from_file(
    "experiments/results/experiment_circuits/input_circuits/"
    f"{circuit_name}.qasm"
)
p = (-3, 1)
region = circuit.surround(point=p, num_qudits=4)
print(region)
# print(circuit[p])
