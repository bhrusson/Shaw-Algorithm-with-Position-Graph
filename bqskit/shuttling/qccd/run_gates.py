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
from bqskit.shuttling.qccd.mapping import QCCDPAMLayoutPass, QCCDPAMRoutingPass, QCCDLayoutPass, QCCDRoutingPass
from bqskit.shuttling.QCCD_schedule_new import schedule_qccd_from_instructions_v3
from bqskit.shuttling.qccd import (QCCDMachineModel, QCCDSubtopologySelectionPass, create_grid_physical_machine,
                                   QCCDMappingAlgorithm, create_testing_physical_machine, schedule_QCCD, schedule_QCCD_w_fid)
import sys
import pickle
from pathlib import Path
#enable_logging(True)
input_filename = sys.argv[1]
algo_type = sys.argv[2]
trap_type = sys.argv[3]
trap_capacity = int(sys.argv[4])
num_layout_passes = int(sys.argv[5])
gate_type = sys.argv[6]
if len(sys.argv) < 8:
    run_index = 0
else:
    run_index = sys.argv[7]
if len(sys.argv) < 9:
    seed = 1234
else:
    seed = int(sys.argv[8])
if len(sys.argv) < 10:
    routing_mode = 'bruteforce'
else:
    routing_mode = sys.argv[9]
if routing_mode not in ('bruteforce', 'heuristic'):
    raise ValueError("routing_mode must be 'bruteforce' or 'heuristic'.")
# qasm_result_filename = sys.argv[5]
# result_filename = sys.argv[6]
print("Input filename: ", str(input_filename))
print("Algorithm: ", str(algo_type))
print("Trap type: ", str(trap_type))
print("Trap capacity: ", str(trap_capacity))
print("Num layout passes: ", str(num_layout_passes))
print("Gate type: ", str(gate_type))
print("Run index: ", str(run_index))
print("Seed: ", str(seed))
print("Routing mode: ", str(routing_mode))
# print("QASM output filename: ", str(qasm_result_filename))
# print("Output filename: ", str(result_filename))
if trap_type == "grid":
    physical_model = create_grid_physical_machine(num_cols = 1,
                                                  num_rows = 1,
                                                  trap_capacity = trap_capacity)
else:
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
# timing_data = {'sq_timings': 40e-6,
#                'tq_timings': 40e-6,
#                'segment': 40e-6,
#                'inner_swap': 40e-6,
#                'split': 80e-6,
#                'merge': 80e-6,
#                'junction_Y': 120e-6,
#                'junction_X': 120e-6}
gate_set = GateSet({U3Gate(), CXGate()})
machine_model = QCCDMachineModel(gate_set=gate_set,
                                 physical_graph=physical_model,
                                 multi_qudit_gate_type=gate_type,
                                 timing_data=timing_data)
print("Position graph: ", machine_model.position_graph)
benchmark_dir = Path("bqskit/shuttling/qccd/benchmark_circuits")
output_dir = Path("bqskit/shuttling/qccd/paper_result_gate")
output_dir.mkdir(parents=True, exist_ok=True)
cir = Circuit.from_file(str(benchmark_dir / f"{input_filename}.qasm"))
target_unitary = cir
num_qudits = cir.num_qudits

"""
    Define ion assignment
"""
# initial_assignment = {0: [13, 14, 15, 16, 17], 1: [12, 11, 10, 9, 8],
#     2: [], 3: [18, 19, 20, 21, 22], 4: [7, 28, 6, 5, 4],
#     5: [31, 0], 6: [23, 24, 25, 26, 27], 7: [29, 3, 30, 2, 1], 8: []}

"""
QFT_wsq_32
    {0: [21, 10, 30, 26, 1], 1: [28, 3, 27, 4, 25],
    2: [22, 9, 20, 11, 19], 3: [5, 31, 0, 29, 2],
    4: [6, 24, 7, 23, 8], 5: [12, 18, 13, 17, 14],
    6: [], 7: [16, 15], 8: []}
TFIM_wsq_n32_s100
   {0: [12, 13, 14, 15, 16],
   1: [11, 10, 9, 8, 7], 2: [],
   3: [17, 18, 19, 20, 21], 4: [27, 28, 6, 5, 29],
   5: [1, 0], 6: [22, 23, 24, 25, 26],
   7: [4, 3, 30, 2, 31], 8: []}
TFXY_wsq_n32_s100
    {0: [13, 14, 15, 16, 17], 1: [12, 11, 10, 9, 8],
    2: [], 3: [18, 19, 20, 21, 22], 4: [7, 28, 6, 5, 4],
    5: [31, 0], 6: [23, 24, 25, 26, 27], 7: [29, 3, 30, 2, 1], 8: []}
"""

ion_assignment = {}
####
all_available_trap_space = []
for trap in machine_model.physical_graph.trap_list:
    all_available_trap_space += machine_model.physical_to_position[trap.id]
random.seed(seed)
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
    congestion_rate = 1.0
else:
    congestion_rate = 0.5
print("Congestion rate: ", congestion_rate)

gate_count_weight = 0.1
force_bruteforce = routing_mode == 'bruteforce'

qsearch_pass = QSearchSynthesisPass()
block_size = 3
if algo_type == "SHAW":
    workflow = [
        UnfoldPass(),
        SetModelPass(machine_model),
        UpdateDataPass(key='ion_assignment_qccd', val=ion_assignment),
        # Re-target the gate
        QuickPartitioner(block_size),
        ApplyPlacement(),
        QCCDLayoutPass(total_passes=num_layout_passes,
                       cogestion_rate=congestion_rate,
                       force_bruteforce=force_bruteforce),
        QCCDRoutingPass(gate_count_weight,
                        cogestion_rate=congestion_rate,
                        force_bruteforce=force_bruteforce),
        ApplyPlacement(),
        UnfoldPass(),
    ]
elif algo_type == "SHAPER":
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
                                     vary_topology=True),
            ScanningGateRemovalPass(),
        ),
        UnfoldPass(),
    ]
else:
    raise ValueError("algo_type must be SHAW or SHAPER")

with Compiler() as compiler:
    start = timer()
    output_circuit, data = compiler.compile(target_unitary, workflow, request_data=True)
    end = timer()
    compile_time = end - start

"""
Save qasm file
"""
qasm_result_filename = output_dir / (
    f"{algo_type}_{input_filename}_idx{run_index}_{trap_type}_{trap_capacity}_{num_layout_passes}.qasm"
)
output_circuit.save(str(qasm_result_filename))

schedule_result = schedule_qccd_from_instructions_v3(
    instruction_lst=data["instruction_list"],
    initial_ion_assignment=data['initial_ion_assignment_qccd'],
    machine_model=data.model,
    circuit=output_circuit,
    parallel=True,
)

print(f"Compile time (s): {compile_time}")
print(f"Runtime (us): {schedule_result['runtime'] / 1e-6}")
print(f"Application fidelity: {schedule_result['application_fidelity']}")
print(f"Instruction count: {len(data['instruction_list'])}")
print(f"Execute rounds: {len(schedule_result['execute_rounds'])}")
print(f"Move rounds: {len(schedule_result['move_rounds'])}")
