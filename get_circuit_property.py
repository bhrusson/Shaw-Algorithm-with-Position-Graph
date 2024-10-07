from bqskit import Circuit

circuit_lst = ["toffoli.qasm", "fredkin.qasm", "hubbard_4_0.qasm", "Grover_5.qasm",
               "PhaseEstimator_5.qasm", "Grover_8.qasm", "PhaseEstimator_8.qasm",
               "hubbard_2x2_0.qasm", "adder9.qasm", "hubbard_3x2_0.qasm",
               "cuccaro-adder-16.qasm", "heisenberg-16-20.qasm", "incrementer-16.qasm",
               "QAOA_16.qasm", "QFT_16.qasm", "QuantumVolume_16.qasm", "TFIM_n16_s100.qasm",
               "TFXY_n16_s100.qasm", "QAOA_20.qasm", "QFT_20.qasm", "QuantumVolume_20.qasm"]

for i in circuit_lst:
    print(i)
    circuit = Circuit.from_file("./experiments/results/experiment_circuits/"
                                "input_circuits/" + i)
    num_cycle = circuit.num_cycles
    print("Number of circuit cycles: ", num_cycle)
    num_tq_gate = 0
    for layer_idx in range(num_cycle):
        layer = circuit[layer_idx]
        for op in layer:
            if op.num_qudits == 2:
                num_tq_gate += 1
    print("Number of tq gates:", num_tq_gate)