from __future__ import annotations
import logging

from bqskit import Circuit
from bqskit.compiler import PassData
from bqskit.ir.gates import U3Gate, RZZGate, SwapGate
from bqskit.passes import LayerGenerator
from bqskit.qis import UnitaryMatrix, StateVector, StateSystem

_logger = logging.getLogger(__name__)


class ShuttlingLayerGenerator(LayerGenerator):
    key = "__ShuttlingLayerGenerator_gatezone"

    def retrieve_gatezones(self, num_qudits: int, data: PassData):
        if self.key not in data:
            gate_zones = set(i for i in range(0, num_qudits, 2))
        else:
            gate_zones = data[self.key]

        cg = data.connectivity
        if len(gate_zones) > 1:
            tq_zones = gate_zones
            sq_zones = []
            for i in gate_zones:
                sq_zones.append(i)
                for neighbor in cg.get_neighbors_of(i):
                    sq_zones.append(neighbor)
            sq_zones = set(sq_zones)
        else:
            tq_zone_value = list(gate_zones)[0]
            tq_zones = {int(tq_zone_value)}
            # sq_zones = {int(tq_zone_value), int(np.round((tq_zone_value - int(tq_zone_value))*10))}
            if num_qudits == 3:
                sq_zones = {cg.get_neighbors_of(tq_zone_value)[0], cg.get_neighbors_of(tq_zone_value)[1]}
            elif num_qudits == 2:
                sq_zones = {cg.get_neighbors_of(tq_zone_value)[0]}
            else:
                raise ValueError("Invalid number of qudits")
            # _logger.debug(f"sq_zones: {sq_zones}")
        # _logger.debug(f'Retrive gate zone successfully. Single qubit zone: {sq_zones}; Two qubit zones: {tq_zones}.')
        return sq_zones, tq_zones

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

        sq_zones, tq_zones = self.retrieve_gatezones(target.num_qudits, data)
        initial_circuit = Circuit(target.num_qudits)

        for i in sq_zones:
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
        sq_zones, tq_zones = self.retrieve_gatezones(circuit.num_qudits, data)
        # Generate successors
        successors = []

        # Computational gate generation
        if len(tq_zones) > 1:
            for i in tq_zones:
                if i >= circuit.num_qudits:
                    continue
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [i, cg.get_neighbors_of(i)[0]])
                successor.append_gate(U3Gate(), [i])
                successor.append_gate(U3Gate(), [cg.get_neighbors_of(i)[0]])

                successors.append(successor)
        else:
            tq_zone_val = list(tq_zones)[0]
            for i in cg.get_neighbors_of(tq_zone_val):
                successor = circuit.copy()
                successor.append_gate(RZZGate(), [tq_zone_val, i])
                successor.append_gate(U3Gate(), [tq_zone_val])
                successor.append_gate(U3Gate(), [i])
                successors.append(successor)

        # Shuttling generation
        # for edge in cg:
        #     if edge[0] in sq_zones and edge[1] in sq_zones:
        #         continue
        #     if not circuit.is_point_idle((-1, edge[0])):
        #         op = circuit[-1, edge[0]]
        #         if op.location == edge and op.gate == SwapGate():
        #             continue
        #     successor = circuit.copy()
        #     successor.append_gate(SwapGate(), location=edge)
        #     if edge[0] in sq_zones:
        #         successor.append_gate(U3Gate(), [edge[0]])
        #     if edge[1] in sq_zones:
        #         successor.append_gate(U3Gate(), [edge[1]])
        #     successors.append(successor)
        return successors
