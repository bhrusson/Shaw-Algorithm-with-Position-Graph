import json
import numpy as np
from pytket.phir.api import pytket_to_phir
from pytket.phir.qtm_machine import QTM_MACHINES_MAP, QTM_DEFAULT_GATESET
from pytket.phir.machine import Machine, MachineTimings
from pytket.phir.sharding.sharder import Sharder
from pytket.phir.place_and_route import place_and_route
from pytket.phir.phirgen_parallel import genphir_parallel
from bqskit import Circuit
from bqskit.ext import bqskit_to_pytket
from pytket.phir.qtm_machine import QtmMachine

def construct_qtm_machine(num_qudits: int):
    tq_options = set()
    for tq in range(0, num_qudits, 2):
        tq_options.add(tq)
    QTM_MACHINES_MAP[QtmMachine.H1] = Machine(
        size=num_qudits,
        gateset=QTM_DEFAULT_GATESET,
        tq_options=tq_options,
        timings=MachineTimings(
            tq_time=0.04,
            sq_time=0.03,
            qb_swap_time=0.9,
            meas_prep_time=0.05,
        ),
    )

num_qudits = 3
circuit_type = "toffoli"
cir = Circuit.from_file(f"experiments/results/experiment_circuits"
                        f"/input_circuits/{circuit_type}.qasm")
pytket_circuit = bqskit_to_pytket(cir)

construct_qtm_machine(4)
phir_json = pytket_to_phir(circuit=pytket_circuit, qtm_machine=QtmMachine.H1)
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
print("Operation list: ", qop_lst)
print("Counting: ", np.unique(qop_lst, return_counts=True))
print("Total duration: ", total_duration)
