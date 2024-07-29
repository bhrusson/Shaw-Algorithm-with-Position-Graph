import json
import numpy as np
from bqskit import Circuit
from pytket.phir.rebasing.rebaser import rebase_to_qtm_machine
from pytket.phir.qtm_machine import QtmMachine
from bqskit.shuttling.util import get_duration_from_circ
from bqskit.ext import bqskit_to_pytket, pytket_to_bqskit

num_qudits = 16
circuit_type = "QFT"
circuit = Circuit.from_file(f"experiments/results/experiment_circuits"
                            f"/input_circuits/{circuit_type}_{num_qudits}.qasm")
pytket_cir = bqskit_to_pytket(circuit)
pytket_cir = rebase_to_qtm_machine(pytket_cir, QtmMachine.H1)
circuit = pytket_to_bqskit(pytket_cir)
print(get_duration_from_circ(circuit, QtmMachine.H1))
