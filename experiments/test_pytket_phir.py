import json
import numpy as np
import pytket.qasm
from numpy import pi
from pytket.phir.api import pytket_to_phir
from pytket import Circuit
from bqskit.ext import bqskit_to_pytket
from pytket.phir.qtm_machine import QtmMachine
from circuit_generator import circuit_generate


# circuit = circuit_generate("Toffoli", 3)

# pytket_circuit = Circuit(3)
# pytket_circuit.H(0)
# pytket_circuit.CU1(pi/2, 1, 0)
# pytket_circuit.CU1(pi/4, 2, 0)
# pytket_circuit.H(1)
# pytket_circuit.CU1(pi/2, 2, 1)
# pytket_circuit.H(2)
# pytket_circuit.SWAP(0, 2)
# pytket_circuit = bqskit_to_pytket(circuit)
circuit_type = "adder9"
pytket_circuit = pytket.qasm.circuit_from_qasm(f"experiments/results/experiment_circuits/input_circuits/{circuit_type}"
                                               ".qasm")
json_phir = pytket_to_phir(circuit=pytket_circuit, qtm_machine=QtmMachine.H1_1)
phir = json.loads(json_phir)
# print(phir)
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
print("Operation list: ", qop_lst)
print("Counting: ", np.unique(qop_lst, return_counts=True))
print("Slash list: ", slash_lst)
print("Total duration: ", total_duration)
