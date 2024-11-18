from bqskit import Circuit, compile, MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.ir.gates import CXGate

cir = Circuit.from_file(f"bqskit/shuttling/qccd/benchmark_circuits/bqskit_QFT_200.qasm")
gateset:set[Gate] = {U3Gate(), CXGate()}
machine_model = MachineModel(200, None, gateset)
compiled_circuit = compile(cir, model=machine_model, optimization_level=3)
compiled_circuit.save(f"bqskit/shuttling/qccd/benchmark_circuits/QFT_200_compiled.qasm")