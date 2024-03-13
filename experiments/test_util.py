from circuit_generator import circuit_generate
from bqskit.shuttling.util import check_executable_circuit
circuit = circuit_generate("QFT", 3, False)
executable = check_executable_circuit(circuit, [0])
print(executable)