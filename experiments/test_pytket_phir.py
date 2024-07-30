import json
import numpy as np
from pytket.phir.api import pytket_to_phir
from pytket.phir.qtm_machine import QTM_MACHINES_MAP
from pytket.phir.sharding.sharder import Sharder
from pytket.phir.place_and_route import place_and_route
from pytket.phir.phirgen_parallel import genphir_parallel
from pytket.phir.phirgen import genphir
from bqskit import Circuit
from bqskit.ext import bqskit_to_pytket
from pytket.phir.qtm_machine import QtmMachine

num_qudits = 4
circuit_type = "hubbard_4_0"
cir = Circuit.from_file(f"experiments/results/experiment_circuits"
                        f"/input_circuits/{circuit_type}.qasm")
pytket_circuit = bqskit_to_pytket(cir)
phir_json = pytket_to_phir(circuit=pytket_circuit, qtm_machine=QtmMachine.H1)

# machine = QTM_MACHINES_MAP.get(QtmMachine.H1)
# shards = Sharder(pytket_circuit).shard()
# placed = place_and_route(shards, machine)
# phir_json = genphir_parallel(placed, machine)

phir = json.loads(phir_json)
total_duration = 0
qop_lst = []
slash_lst = []
for i in phir['ops']:
    print(i)
    if 'qop' in i.keys():
        qop_lst.append(i['qop'])
    elif '//' in i.keys():
        slash_lst.append(i['//'])
    elif 'block' in i.keys():
        for j in i['ops']:
            qop_lst.append(j['qop'])
    elif 'mop' in i.keys():
        total_duration += i['duration'][0]
# print("Operation list: ", qop_lst)
# print("Counting: ", np.unique(qop_lst, return_counts=True))
print("Total duration: ", total_duration)
