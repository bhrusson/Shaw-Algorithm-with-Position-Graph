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

ion_assignment = {0: 35, 1: 6, 2: 17, 3: 8, 4: 1, 5: 34, 6: 5, 7: 15, 8: 16, 9: 0,
                  10: 37, 11: 12, 12: 11, 13: 13, 14: 10, 15: 7, 16: 9, 17: 33, 18: 24, 19: 39}
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
    gate=Operation(CXGate(), (9, 18)),
    D=machine_model.all_pair_travelling_time(),
    pi=pi,
    ion_assignment=ion_assignment
)
