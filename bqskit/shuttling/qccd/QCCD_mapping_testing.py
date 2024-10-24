from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate

physical_model = create_testing_physical_machine()
timing_data = {'sq_timings': 30e-6,
               'tq_timings': 40e-6,
               'segment': 5e-6,
               'inner_swap': 42e-6,
               'split': 80e-6,
               'merge': 80e-6,
               'junction_Y': 100e-6,
               'junction_X': 120e-6}
machine_model = QCCDMachineModel(physical_graph=physical_model,
                                 timing_data=timing_data)
ion_assignment = {0: 0, 1: 1, 2: 12, 3: 14}
circuit = Circuit(4)
circuit.append_gate(CNOTGate(), (2, 3))
mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                    extended_set_size=5,
                                    extended_set_weight=0.01)
mapping_algo.forward_pass(circuit, ion_assignment)
