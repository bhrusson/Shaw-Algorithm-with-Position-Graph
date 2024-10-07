from experiments.prepare_running import *
from timeit import default_timer as timer
from bqskit.shuttling.util import get_duration_from_circ
import sys
import pickle

input_filename = sys.argv[1]
qasm_result_filename = sys.argv[2]
result_filename = sys.argv[3]
print("Input filename: ", str(input_filename))
print("QASM output filename: ", str(qasm_result_filename))
print("Output filename: ", str(result_filename))

cir = Circuit.from_file(input_filename)
target_unitary = cir

with Compiler() as compiler:
    start = timer()
    output_circuit, data = compiler.compile(target_unitary, workflow, request_data=True)
    end = timer()
    compile_time = end - start

"""
Save qasm file
"""
output_circuit.save(qasm_result_filename)


"""
Save pickle result file
"""
duration = get_duration_from_circ(output_circuit, qtm_machine.H1)
result = [
          duration, compile_time,
          output_circuit.gate_counts,
          data['initial_mapping'],
          data['final_mapping']
          ]
with open(result_filename, 'wb') as f:
    pickle.dump(result, f)
