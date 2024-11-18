from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CCXGate, CXGate
from bqskit.ir import Operation

physical_model = create_testing_physical_machine(trap_capacity=5,
                                                 type='Enchilada')
timing_data = {'sq_timings': 30e-6,
               'tq_timings': 40e-6,
               'segment': 5e-6,
               'inner_swap': 42e-6,
               'split': 80e-6,
               'merge': 80e-6,
               'junction_Y': 100e-6,
               'junction_X': 120e-6}
machine_model = QCCDMachineModel(physical_graph=physical_model,
                                 multi_qudit_gate_type='FM',
                                 timing_data=timing_data)
# [8, 11] with
ion_assignment = {0: 4, 1: 33, 2: 42, 3: 38, 4: 8, 5: 11, 6: 35, 7: 12, 8: 14, 9: 16, 10: 32, 11: 15, 12: 2, 13: 7, 14: 39, 15: 40, 16: 41, 17: 44, 18: 45, 19: 17}
# pi = [10, 7, 9, 8, 3, 2, 5, 1, 19, 0, 18, 4, 13, 6, 14, 15, 17, 12, 11, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
pi = list(range(machine_model.num_qudits))

# circuit = Circuit(8)
# circuit.append_gate(CCXGate(), (1, 3, 6))
# machine_model.update_wrt_perm(initial_placement=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
#                               permutation=pi)
print(machine_model.position_graph)
print(machine_model.physical_to_position)
print(machine_model.segment_assignment)
mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                    cogestion_rate=0.8,
                                    decay_delta=0.00,
                                    extended_set_size=5,
                                    extended_set_weight=0.5)
mapping_algo._brute_force_congestion(
    gate=Operation(CCXGate(), (19, 13, 17)),
    D=machine_model.all_pair_travelling_time(),
    pi=pi,
    ion_assignment=ion_assignment
)
