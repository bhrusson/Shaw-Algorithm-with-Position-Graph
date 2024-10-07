from bqskit.passes import *
from bqskit.ir import Circuit
from bqskit.compiler import Compiler
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_layergen import QCCDLayerGenerator
from bqskit.shuttling.qccd.QCCD_heuristic_search import QCCDHeuristicFunction
from bqskit import enable_logging

enable_logging(True)
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
ion_assignment = {0: 0, 1: 2, 2: 6}

qsearch_pass = QSearchSynthesisPass(layer_generator=QCCDLayerGenerator(),
                                    heuristic_function=QCCDHeuristicFunction(heuristic_factor=5,
                                                                             machine_model=machine_model,
                                                                             ion_assignment=ion_assignment))
block_size = 3
workflow = [
    qsearch_pass
]
test_circuit = 'toffoli.qasm'
input_filename = "experiments/results/experiment_circuits/input_circuits/" + test_circuit
cir = Circuit.from_file(input_filename)
with Compiler() as compiler:
    output_circuit, data = compiler.compile(cir, workflow, request_data=True)

"""
Save qasm file
"""
qasm_result_filename = "experiments/results/experiment_circuits/QCCD_output_circuits/" + test_circuit
output_circuit.save(qasm_result_filename)

