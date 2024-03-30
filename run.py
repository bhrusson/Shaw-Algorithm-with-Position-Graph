from experiments.prepare_running import *

with Compiler() as compiler:
    output_circuit, data = compiler.compile(target_circuit, workflow, request_data=True)

print("Circuit is executable: ", check_executable_circuit(output_circuit, machine.tq_options))
print(output_circuit.gate_counts)
print(output_circuit.coupling_graph)
# print("Distance from correct unitary: ", output_circuit.get_unitary().get_distance_from(target, 1))
print("QASM: ")
print(output_circuit.to("qasm"))
output_circuit.save(f"experiments/results/experiment_circuits/output_circuits/{circuit_type}.qasm")
duration, num_shifts = get_duration_from_circ_after_scheduling(output_circuit, qtm_machine.H1_1)
print(f"Duration of circuit: {duration} with {num_shifts} shift gates ")
print(f"Initial mapping of circuit: {data['initial_mapping']}")
print(f"Final mapping of circuit: {data['final_mapping']}")
# if data['final_mapping'] != [0, 1, 2]:
#     output_circuit.append_gate(PermutationGate(3, data['final_mapping']), [0, 1, 2])
# print("Distance from correct unitary after applying permutation: ",
#       output_circuit.get_unitary().get_distance_from(target, 1))
