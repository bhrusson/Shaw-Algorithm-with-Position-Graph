import bqskit
from pytket.circuit import Circuit
from pytket.phir.rebasing.rebaser import rebase_to_qtm_machine
from pytket.phir.qtm_machine import QtmMachine
from bqskit.ext import bqskit_to_pytket

num_qudits = 16
circuit_type = "QFT"
circuit = bqskit.Circuit.from_file(f"experiments/results/experiment_circuits"
                            f"/input_circuits/{circuit_type}_{num_qudits}_bqskit_compiled.qasm")
pytket_cir = bqskit_to_pytket(circuit)
pytket_cir = rebase_to_qtm_machine(pytket_cir, QtmMachine.H1)
print(get_duration_from_circ_pytket(pytket_cir, QtmMachine.H1))
