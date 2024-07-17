from __future__ import annotations
import logging

from bqskit import Circuit
from bqskit.compiler import PassData
from bqskit.ir.gates import U3Gate, RZZGate, SwapGate
from bqskit.passes import LayerGenerator
from bqskit.qis import UnitaryMatrix, StateVector, StateSystem

_logger = logging.getLogger(__name__)


class ShuttlingShiftGenerator(LayerGenerator):

    def __init__(self, machine_state):
        self.machine_state = machine_state

    def gen_initial_layer(
            self,
            target: UnitaryMatrix | StateVector | StateSystem,
            data: PassData,
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` is not qubit only.
        """
        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not target.is_qubit_only():
            raise ValueError('Cannot generate layers for non-qubit circuits.')

        initial_circuit = Circuit(target.num_qudits)

        for i in range(initial_circuit.num_qudits):
            initial_circuit.append_gate(U3Gate(), [i])
        return initial_circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Generate successors
        successors = []

        # Computational gate generation
        if self.machine_state is False:
            for i in range(0, circuit.num_qudits-1, 2):
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [i, i + 1])
                successor.append_gate(U3Gate(), [i])
                successor.append_gate(U3Gate(), [i + 1])
                successors.append(successor)
            successor = circuit.copy()
            for i in range(circuit.num_qudits):
                successor.append_gate(U3Gate(), [i])
            successors.append(successor)
        else:
            for i in range(1, circuit.num_qudits-1, 2):
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [i, i + 1])
                successor.append_gate(U3Gate(), [i])
                successor.append_gate(U3Gate(), [i + 1])
                successors.append(successor)
            successor = circuit.copy()
            for i in range(circuit.num_qudits):
                successor.append_gate(U3Gate(), [i])
            successors.append(successor)
        return successors
