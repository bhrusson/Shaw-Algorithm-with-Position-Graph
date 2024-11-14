from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CCXGate
from bqskit.ir import Operation

physical_model = create_testing_physical_machine(trap_capacity = 6,
                                                 type = 'H')
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
ion_assignment = {0: 24, 1: 10, 2: 23, 3: 8, 4: 1, 5: 2, 6: 20, 7: 13, 8: 25, 9: 28, 10: 21, 11: 22, 12: 18, 13: 3, 14: 15, 15: 4, 16: 9, 17: 11, 18: 7, 19: 5}
pi = [10, 7, 9, 8, 3, 2, 5, 1, 19, 0, 18, 4, 13, 6, 14, 15, 17, 12, 11, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
# circuit = Circuit(8)
# circuit.append_gate(CCXGate(), (1, 3, 6))
machine_model.update_wrt_perm(initial_placement=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                              permutation=pi)
print(machine_model.position_graph)
print(machine_model.physical_to_position)
mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                    cogestion_rate=0.9,
                                    decay_delta=0.00,
                                    extended_set_size=5,
                                    extended_set_weight=0.5)
mapping_algo._brute_force_congestion(
    gate=Operation(CCXGate(), (pi.index(13), pi.index(14), pi.index(15))),
    D=machine_model.all_pair_travelling_time(),
    pi=pi,
    ion_assignment=ion_assignment
)
