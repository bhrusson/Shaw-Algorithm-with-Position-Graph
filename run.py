from experiments.prepare_running import *

with Compiler() as compiler:
    output_circuit, data = compiler.compile(circuit, workflow, request_data=True)

print("Circuit is executable: ", check_executable_circuit(output_circuit, machine.tq_options))
print(output_circuit.gate_counts)
print(output_circuit.coupling_graph)
print("Distance from correct unitary: ", output_circuit.get_unitary().get_distance_from(target, 1))
print("QASM: ")
print(output_circuit.to("qasm"))
# output_circuit.save("qv.qasm")
print(f"Duration of circuit: {get_duration_from_circ(output_circuit, qtm_machine)}")
print(f"Initial mapping of circuit: {data['initial_mapping']}")
print(f"Final mapping of circuit: {data['final_mapping']}")
if data['final_mapping'] != [0, 1, 2]:
    output_circuit.append_gate(PermutationGate(3, data['final_mapping']), [0, 1, 2])
print("Distance from correct unitary after applying permutation: ",
      output_circuit.get_unitary().get_distance_from(target, 1))
