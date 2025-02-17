import sys
from bqskit import Circuit, compile

input_filename = sys.argv[1]
openqasm_file_name = f"bqskit/shuttling/qccd/benchmark_circuits/{input_filename}.qasm"
cir = Circuit.from_file(openqasm_file_name)

compiled_circuit = compile(cir, optimization_level=4)
compiled_circuit.save(f"bqskit/shuttling/qccd/benchmark_circuits/{input_filename}_precompiled_alltoall.qasm")