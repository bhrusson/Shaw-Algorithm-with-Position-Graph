from timeit import default_timer as timer
import random
from bqskit.passes import *
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
import sys
import pickle
enable_logging(True)
input_filename = sys.argv[1]
trap_type = sys.argv[2]
trap_capacity = int(sys.argv[3])
num_layout_passes = int(sys.argv[4])
gate_type = sys.argv[5]
# qasm_result_filename = sys.argv[5]
# result_filename = sys.argv[6]
print("Input filename: ", str(input_filename))
print("Trap type: ", str(trap_type))
print("Trap capacity: ", str(trap_capacity))
print("Num layout passes: ", str(num_layout_passes))
print("Gate type: ", str(gate_type))
# print("QASM output filename: ", str(qasm_result_filename))
# print("Output filename: ", str(result_filename))

physical_model = create_testing_physical_machine(type=trap_type,
                                                 trap_capacity=int(trap_capacity))
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
                                 multi_qudit_gate_type=gate_type,
                                 timing_data=timing_data)

cir = Circuit.from_file(f"bqskit/shuttling/qccd/benchmark_circuits/{input_filename}.qasm")
target_unitary = cir
num_qudits = cir.num_qudits

"""
    Define ion assignment
"""
ion_assignment = {}
all_available_trap_space = []
for trap in machine_model.physical_graph.executable_trap_list:
    all_available_trap_space += machine_model.physical_to_position[trap.id]
trap_seq = random.sample(all_available_trap_space, num_qudits)
for i in range(num_qudits):
    ion_assignment[i] = trap_seq[i]
print("Initial ion assignment: ", ion_assignment)

"""
    Define congestion rate
"""
executable_spaces = 0
for trap in machine_model.physical_graph.executable_trap_list:
    executable_spaces += trap.max_num_ions
if cir.num_qudits == executable_spaces:
    congestion_rate = 0.6
else:
    congestion_rate = 1.0
print("Congestion rate: ", congestion_rate)

gate_count_weight = 0.1

qsearch_pass = QSearchSynthesisPass()
block_size = 3
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
    QCCDPAMLayoutPass(total_passes=num_layout_passes,
                      cogestion_segment_rate=congestion_rate),
    QCCDPAMRoutingPass(gate_count_weight,
                       cogestion_segment_rate=congestion_rate),
    ApplyPlacement(),
    UnfoldPass(),
]

with Compiler() as compiler:
    start = timer()
    output_circuit, data = compiler.compile(target_unitary, workflow, request_data=True)
    end = timer()
    compile_time = end - start

"""
Save qasm file
"""
qasm_result_filename = f"bqskit/shuttling/qccd/new_result/{input_filename}_{trap_type}_{trap_capacity}_{num_layout_passes}.qasm"
output_circuit.save(qasm_result_filename)

"""
Calculating runtime
"""
runtime = schedule_QCCD(data["instruction_list"],
                        output_circuit,
                        data.initial_mapping,
                        data['initial_ion_assignment_qccd'],
                        data.model,
                        parallel=True)
print(f"Scheduling QCCD: {runtime / 1e-6} us")

"""
Save pickle result file
"""
result = [
          runtime, compile_time,
          data["instruction_list"],
          output_circuit,
          output_circuit.gate_counts,
          data['initial_ion_assignment_qccd'],
          data['initial_mapping'],
          data['final_mapping'],
          machine_model
          ]
result_filename = f"bqskit/shuttling/qccd/new_result/{input_filename}_{trap_type}_{trap_capacity}_{num_layout_passes}.pkl"
with open(result_filename, 'wb') as f:
    pickle.dump(result, f)
