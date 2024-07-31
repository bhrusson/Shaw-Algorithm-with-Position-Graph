import sys
import json
import pickle
import numpy as np
from bqskit import Circuit
from bqskit.ext import bqskit_to_pytket
from pytket.phir.api import pytket_to_phir
from pytket.phir.qtm_machine import QtmMachine


input_filename = sys.argv[1]
json_filename = sys.argv[2]
result_filename = sys.argv[3]
print("Input filename: ", str(input_filename))
print("Json filename: ", str(json_filename))
print("Output filename: ", str(result_filename))
"""
Run the pytket-phir estimation
"""
cir = Circuit.from_file(input_filename)
pytket_circuit = bqskit_to_pytket(cir)
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
