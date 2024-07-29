from experiments.prepare_running import *

with Compiler() as compiler:
    output_circuit, data = compiler.compile(target_unitary, workflow, request_data=True)

print("Circuit is executable: ", check_executable_circuit(output_circuit, machine.tq_options))
print(output_circuit.gate_counts)
print(output_circuit.coupling_graph)
output_circuit.save(f"experiments/results/experiment_circuits/"
                    f"output_circuits/{circuit_type}_{num_qudits}_wo_scheduling.qasm")
duration = get_duration_from_circ(output_circuit, qtm_machine.H1)
print(f"Duration of circuit: {duration}") # with {num_shifts} shift gates ")
print(f"Initial mapping of circuit: {data['initial_mapping']}")
print(f"Final mapping of circuit: {data['final_mapping']}")
