from __future__ import annotations

from bqskit import Circuit
from bqskit.compiler import PassData
from bqskit.ir.gates import U3Gate, RZZGate, SwapGate
from bqskit.passes import LayerGenerator
from bqskit.qis import UnitaryMatrix, StateVector, StateSystem


class ShuttlingLayerGenerator(LayerGenerator):
    def __init__(self, gate_zones: set[int]):
        unique_gate_zones = set(gate_zones)
        if len(unique_gate_zones) != len(gate_zones):
            raise ValueError(f"Duplicate gate zone detected: {gate_zones=}")

        for idx, gate_zone in enumerate(gate_zones):
            if gate_zone + 1 in gate_zones:
                raise ValueError(f"Invalid gate zone at index {idx}, gate zone {gate_zone + 1} is described twice.")

        self.tq_zones = gate_zones
        self.sq_zones = []
        for i in gate_zones:
            self.sq_zones.append(i)
            self.sq_zones.append(i + 1)

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

        for i in self.sq_zones:
            if i >= target.num_qudits:
                continue
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

        cg = data.connectivity
        # Generate successors
        successors = []

        # Computational gate generation
        for i in self.tq_zones:
            if i >= circuit.num_qudits:
                continue
            if i != circuit.num_qudits - 1:
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [i, i + 1])
                successor.append_gate(U3Gate(), [i])
                successor.append_gate(U3Gate(), [i + 1])
                successor.append_gate(RZZGate(), [i, i + 1])
                successors.append(successor)

            if i != 0:
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [i - 1, i])
                successor.append_gate(U3Gate(), [i])
                successors.append(successor)
                successor.append_gate(RZZGate(), [i - 1, i])

            # if i != circuit.num_qudits - 1 and i != 0:
            #     successor = circuit.copy()
            #     successor.append_gate(RZZGate(), [i - 1, i])
            #     successor.append_gate(U3Gate(), [i])
            #     successor.append_gate(U3Gate(), [i + 1])
            #     successor.append_gate(RZZGate(), [i, i + 1])
            #     successors.append(successor)

        # Shuttling generation
        for edge in cg:
            if edge[0] in self.sq_zones and edge[1] in self.sq_zones:
                continue
            if not circuit.is_point_idle((-1, edge[0])):
                op = circuit[-1, edge[0]]
                if op.location == edge and op.gate == SwapGate():
                    continue
            successor = circuit.copy()
            successor.append_gate(SwapGate(), location=edge)
            successors.append(successor)

        return successors
