import pickle
import sys
from parse import InputParse
from mappers import *
from machine import Machine, MachineParams, Trap, Segment
from ejf_schedule import Schedule, EJFSchedule
from analyzer import *
from test_machines import *
import numpy as np
import pickle
from timeit import default_timer as timer
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
import matplotlib.pyplot as plt

def print_event(item):
    if item[1] == Schedule.Gate:
        print("GAT", item[4]['ions'], trap_name(item[4]['trap']), (item[2], item[3]))
    elif item[1] == Schedule.Split:
        print("SPL", item[4]['ions'], trap_name(item[4]['trap']) + "->" + seg_name(item[4]['seg']), (item[2], item[3]))
    elif item[1] == Schedule.Move:
        print("MOV", item[4]['ions'], seg_name(item[4]['source_seg']) + "->" + seg_name(item[4]['dest_seg']),
              (item[2], item[3]))
    elif item[1] == Schedule.Merge:
        print("MER", item[4]['ions'], seg_name(item[4]['seg']) + "->" + trap_name(item[4]['trap']), (item[2], item[3]))

circuit_lst = [
    "QAOA_16_compiled",
    "QuantumVolume_16",
    "QFT_16_compiled",
    "TFIM_n16_s100_compiled",
    "TFXY_n16_s100_compiled",
    "QAOA_20_compiled",
    "QuantumVolume_20",
    "QFT_20_compiled"
]

architecture_lst = [
    "H",
    "G2x3"
]

parameter_set = {
    "H": [["4", "5"], ["5", "6"]],
    "G2x3": [["3", "4"], ["4", "5"]],
}

circuit_lst = ["QFT_20_compiled"]
architecture_lst = ["Enchilada"]
parameter_set = {"Enchilada" : ["6"]}
method = "Greedy"
for circuit_idx in range(len(circuit_lst)):
    for architecture in architecture_lst:
        parameter = parameter_set[architecture][0] if circuit_idx < 5 else parameter_set[architecture][1]
        for param_idx in range(len(parameter)):
            param = parameter[param_idx]
            # if param_idx == 0:
            #     continue
            with open(
                    f"bqskit/shuttling/qccd/new_result/QCCDSim_{circuit_lst[circuit_idx]}_{architecture}_{param}_{method}.pkl","rb") as input_file:
                data = pickle.load(input_file)
            schedule = data[1][4]
            init_map = data[2]
            max_ion = int(param)
            print("Initial map", init_map)
            mapping = copy.copy(init_map)
            ion_per_trap = {}
            occupied_seg = []
            for trap in init_map.keys():
                ion_per_trap[trap] = len(init_map[trap])
            for event in schedule.events:
                #print_event(event)
                if event[1] == Schedule.Split:
                    ions = event[4]['ions'][0]
                    trap = event[4]['trap']
                    seg = event[4]['seg']
                    occupied_seg.append(seg)
                    mapping[trap].remove(ions)
                    ion_per_trap[trap] -= 1
                elif event[1] == Schedule.Merge:
                    ions = event[4]['ions'][0]
                    trap = event[4]['trap']
                    seg = event[4]['seg']
                    occupied_seg.remove(seg)
                    mapping[trap].append(ions)
                    ion_per_trap[trap] += 1
                elif event[1] == Schedule.Move:
                    ions = event[4]['ions'][0]
                    seg1 = event[4]['source_seg']
                    seg2 = event[4]['dest_seg']
                    occupied_seg.remove(seg1)
                    occupied_seg.append(seg2)
                else:
                    continue
                assert len(occupied_seg) == len(set(occupied_seg))
                assert np.max(list(ion_per_trap.values())) <= int(max_ion)
    # print("Mapping: ", mapping)
    # print("Occupied segment: ", occupied_seg)


#schedule.print_events()
