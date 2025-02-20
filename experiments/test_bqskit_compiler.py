from bqskit import Circuit, compile, MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.ir.gates import CXGate

n = 50
circuit_type = "QAOA"
cir = Circuit.from_file(f"bqskit/shuttling/qccd/benchmark_circuits/bqskit_{circuit_type}_{n}.qasm")
gateset:set[Gate] = {U3Gate(), CXGate()}
machine_model = MachineModel(n, None, gateset)
compiled_circuit = compile(cir, model=machine_model, optimization_level=3)
compiled_circuit.save(f"bqskit/shuttling/qccd/benchmark_circuits/{circuit_type}_{n}_compiled.qasm")