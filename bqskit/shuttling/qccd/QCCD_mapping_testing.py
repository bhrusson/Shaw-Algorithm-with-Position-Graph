from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.ir.gates import CCXGate, CXGate
from bqskit.ir import Operation

physical_model = create_testing_physical_machine(trap_capacity=6,
                                                 type='H2')
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

ion_assignment = {0: 24, 1: 2, 2: 62, 3: 52, 4: 64, 5: 19, 6: 39, 7: 28, 8: 65, 9: 40, 10: 51, 11: 26, 12: 53, 13: 66, 14: 4
, 15: 14, 16: 7, 17: 55, 18: 3, 19: 61, 20: 16, 21: 38, 22: 15, 23: 29, 24: 27, 25: 9, 26: 8, 27: 56, 28: 18, 29: 46, 30: 10, 31: 30, 32: 54, 33: 33, 34: 63, 35: 11, 36: 13, 37: 36, 38: 43, 39: 20, 40: 5, 41: 21, 42: 45, 43: 58, 44: 50}
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
                                    cogestion_rate=1.0,
                                    decay_delta=0.00,
                                    extended_set_size=5,
                                    extended_set_weight=0.5)
mapping_algo._brute_force_congestion(
    gate=Operation(CCXGate(), (0, 13, 16)),
    D=machine_model.all_pair_travelling_time(),
    pi=pi,
    ion_assignment=ion_assignment
)
