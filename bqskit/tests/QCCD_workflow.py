from bqskit.passes import *
import pickle
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.compiler.gateset import GateSet
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.ir.gates import CXGate
from bqskit import enable_logging
from bqskit.ir.opt import ScipyMinimizer, HilbertSchmidtCostGenerator
from bqskit.shuttling.qccd.mapping import QCCDPAMLayoutPass, QCCDPAMRoutingPass
from bqskit.shuttling.qccd import (QCCDMachineModel, QCCDSubtopologySelectionPass,
                                   QCCDMappingAlgorithm, create_testing_physical_machine, schedule_QCCD
                                  )
from timeit import default_timer as timer

enable_logging(True)
trap_capacity = 5
trap_type = 'G2x3'
gate_count_weight = .1
physical_model = create_testing_physical_machine(type=trap_type,
                                                 trap_capacity=trap_capacity,
                                                 num_traps=6)
timing_data = {'sq_timings': 30e-6,
               'tq_timings': 40e-6,
               'segment': 5e-6,
               'inner_swap': 42e-6,
               'split': 80e-6,
               'merge': 80e-6,
               'junction_Y': 100e-6,
               'junction_X': 120e-6}
gate_set = GateSet({U3Gate(), CXGate()})
machine_model = QCCDMachineModel(gate_set=gate_set,
                                 physical_graph=physical_model,
                                 multi_qudit_gate_type='FM',
                                 timing_data=timing_data)
# This can be found in many ion placement problem
# ion_assignment = {0: 0, 1: 1, 2: 2,
#                   3: 6, 4: 10}

# ion_assignment = {0: 0, 1: 6, 2: 7,
#                   3: 3, 4: 2, 5: 5,
#                   6: 1, 7: 4}

ion_assignment = {0: 0, 1: 1, 2: 2, 3: 3,
                  4: 4, 5: 5, 6: 6, 7: 7,
                  8: 8, 9: 9, 10: 10, 11: 11,
                  12: 12, 13: 13, 14: 14, 15: 15,
                  16: 16, 17: 17, 18: 18, 19: 19}

# sq_synthesis = QSearchSynthesisPass(
#     layer_generator=SingleQuditLayerGenerator(None, allow_repeats=True),
#     heuristic_function=DijkstraHeuristic(),
#     instantiate_options={
#         'method': 'minimization',
#         'minimizer': ScipyMinimizer(),
#         'cost_fn_gen': HilbertSchmidtCostGenerator(),
#     },
# )

qsearch_pass = QSearchSynthesisPass()

block_size = 3
num_layout_passes = 2
workflow = [
    UnfoldPass(),
    SetModelPass(machine_model),
    UpdateDataPass(key='ion_assignment_qccd', val=ion_assignment),
    QCCDSubtopologySelectionPass(block_size),
    # Re-target the gate
    QuickPartitioner(block_size),
    ForEachBlockPass(
        EmbedAllPermutationsPass(inner_synthesis=qsearch_pass,
                                 input_perm=True,
                                 vary_topology=True)
    ),
    ApplyPlacement(),
    QCCDPAMLayoutPass(cogestion_segment_rate=1.),
    QCCDPAMRoutingPass(gate_count_weight,
                       cogestion_segment_rate=1.),
    ApplyPlacement(),
    UnfoldPass(),
    # GroupSingleQuditGatePass(),
    # ForEachBlockPass(
    #     sq_synthesis
    # ),
    # UnfoldPass()
]

file_name = 'QAOA_20_compiled'
input_filename = f"bqskit/shuttling/qccd/benchmark_circuits/{file_name}.qasm"
qasm_result_filename = f"bqskit/shuttling/qccd/result/{file_name}_{trap_type}_{trap_capacity}_{num_layout_passes}.qasm"
cir = Circuit.from_file(input_filename)
target_unitary = cir

with Compiler() as compiler:
    start = timer()
    output_circuit, data = compiler.compile(target_unitary, workflow, request_data=True)
    end = timer()
    compile_time = end - start

print(f"Initial mapping: {data.initial_mapping}")
print(f"Final mapping: {data.final_mapping}")
print(f"Final coupling graph: {data.model.position_graph}")
print(f"Initial ion assignment: {data['initial_ion_assignment_qccd']}")
print(f"Final ion assignment: {data['ion_assignment_qccd']}")
print(f"Moving time from shuttling and inner-swap: {data['moving_time']/1e-6}")
print(f"Instructions list...")
# for instruction in data["instruction_list"]:
#     print(instruction)
"""
    Save data file
"""
full_data = [data["instruction_list"], output_circuit, data.initial_mapping, data['initial_ion_assignment_qccd'], data.model]
with open(f"bqskit/shuttling/qccd/result/{file_name}_{trap_type}_{trap_capacity}_{num_layout_passes}.pkl", "wb") as f:
    pickle.dump(full_data, f, pickle.HIGHEST_PROTOCOL)

"""
    Save qasm file
"""
output_circuit.save(qasm_result_filename)

"""
    Executing runtime
"""
runtime = schedule_QCCD(data["instruction_list"],
                        output_circuit,
                        data.initial_mapping,
                        data['initial_ion_assignment_qccd'],
                        data.model,
                        parallel=True)
print(f"Scheduling QCCD: {runtime / 1e-6} us")


"""
Data for Ed:
    * gate_count_weight: 0.1, extended_set_size = 7, cogestion_segment_rate=.8
      70904.99999999999 us 
    * gate_count_weight: 0.3, extended_set_size = 7, cogestion_segment_rate=.8 (Number of CNOT: 368)
      71102.00000000001 us
    * gate_count_weight: 0.5, extended_set_size = 7, cogestion_segment_rate=.8 (Number of CNOT: 367)
        73583.99999999999 us
    * gate_count_weight: 1.0, extended_set_size = 7, cogestion_segment_rate=.8 (Number of CNOT: 362)
        76635.99999999997 us
    * gate_count_weight: 0.05, extended_set_size = 7, cogestion_segment_rate=.8 (Number of CNOT: 362)
        72868.0 us
"""