from __future__ import annotations

from bqskit import MachineModel
from bqskit.ir.circuit import Circuit


def evaluate_circuit(
    circuit: Circuit,
    machine_model: MachineModel,
    pi: list[int],
    ion_assignment: dict,
) -> float:
    """Estimate circuit execution time under a QCCD ion assignment."""
    runtime = 0.0
    for cycle in range(circuit.num_cycles):
        for gate in circuit[cycle]:
            if gate.num_qudits == 1:
                runtime += machine_model.timing_data['sq_timings']
            elif gate.num_qudits == 2:
                left, right = gate.location
                runtime += machine_model.two_qudit_gate_time(
                    p1=ion_assignment[pi[left]],
                    p2=ion_assignment[pi[right]],
                )
    return runtime
