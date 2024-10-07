from bqskit.shuttling import OddEvenSchedulingPass
from bqskit.shuttling.util import (get_duration_from_circ_after_scheduling,
                                   get_duration_from_circ)

from bqskit.passes import *
from bqskit.ir import Circuit
from bqskit.compiler import Compiler
from pytket.phir.qtm_machine import QtmMachine

workflow = [UnfoldPass(),
            OddEvenSchedulingPass(),
            UnfoldPass()]
input_filename = ("./experiments/results/experiment_circuits/output_circuits/"
                  "QAOA_20_adaption.qasm")
circ = Circuit.from_file(input_filename)
print(f"Duration before scheduling: {get_duration_from_circ(circ, QtmMachine.H1)}")
target_unitary = circ
with Compiler() as compiler:
    output_circuit = compiler.compile(target_unitary, workflow)

print(f"Duration after scheduling: {
    get_duration_from_circ_after_scheduling(output_circuit, QtmMachine.H1)}")