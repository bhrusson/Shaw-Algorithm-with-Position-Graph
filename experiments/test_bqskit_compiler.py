from bqskit import Circuit, compile, MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.ir.gates import CXGate

cir = Circuit.from_file(f"experiments/results/experiment_circuits"
                        f"/input_circuits/Grover_8.qasm")
gateset:set[Gate] = {U3Gate(), CXGate()}
machine_model = MachineModel(20, None, gateset)
compiled_circuit = compile(cir, model=machine_model, optimization_level=3)
compiled_circuit.save(f"experiments/results/experiment_circuits/"
                      f"output_circuits/Grover_8_compiled.qasm")