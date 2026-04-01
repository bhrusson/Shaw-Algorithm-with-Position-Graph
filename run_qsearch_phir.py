import sys
import json
import pickle
import numpy as np
from bqskit import Circuit, compile
from pytket.phir.api import pytket_to_phir
from bqskit.ext import H1_1Model, bqskit_to_pytket
from pytket.phir.qtm_machine import QtmMachine
from run_phir import construct_qtm_machine
from bqskit import enable_logging

enable_logging(True)
input_filename = sys.argv[1]
json_filename = sys.argv[2]
result_filename = sys.argv[3]
print("Input filename: ", str(input_filename))
print("Json filename: ", str(json_filename))
print("Output filename: ", str(result_filename))

"""
Compiling using Qsearch
"""
cir = Circuit.from_file(input_filename)
compiled_circuit = compile(cir, model=H1_1Model, optimization_level=4)

"""
Run the pytket-phir estimation
"""
if cir.num_qudits % 2 == 0:
    construct_qtm_machine(cir.num_qudits)
else:
    construct_qtm_machine(cir.num_qudits + 1)
pytket_circuit = bqskit_to_pytket(compiled_circuit)

phir_json = pytket_to_phir(circuit=pytket_circuit, qtm_machine=QtmMachine.H1)
phir = json.loads(phir_json)
total_duration = 0
qop_lst = []
slash_lst = []
for i in phir['ops']:
    if 'qop' in i.keys():
        qop_lst.append(i['qop'])
    elif '//' in i.keys():
        slash_lst.append(i['//'])
    elif 'block' in i.keys():
        for j in i['ops']:
            qop_lst.append(j['qop'])
    elif 'mop' in i.keys():
        total_duration += i['duration'][0]
counting = np.unique(qop_lst, return_counts=True)

"""
Save json file
"""
with open(json_filename, "w") as outfile:
    outfile.write(phir_json)

"""
Save pickle result file
"""
result = [total_duration, counting]
with open(result_filename, 'wb') as f:
    pickle.dump(result, f)
