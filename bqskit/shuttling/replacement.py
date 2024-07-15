"""This module implements the Dry Scheduling pass."""
from __future__ import annotations

import logging
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import QSearchSynthesisPass
from bqskit.ir import Operation
from bqskit.ir.gates import RZZGate, U3Gate
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from pytket.phir.qtm_machine import QtmMachine
from bqskit.shuttling import ZoneSchedulerPass, ShuttlingShiftGenerator, HeuristicSearch, MachineSchedulingState

_logger = logging.getLogger(__name__)


def alternate_circuit_structure(input_circuit: Circuit, state: MachineSchedulingState) -> Circuit:
    """ Alternate the given circuit to create a temple circuit without the need of shift gate"""
    tmp_circuit = Circuit(num_qudits=input_circuit.num_qudits)
    circ_depth = input_circuit.num_cycles
    for layer_idx in range(circ_depth):
        layer = input_circuit[layer_idx]
        for operation in layer:
            if operation.num_qudits == 1:
                tmp_circuit.append_gate(gate=U3Gate(), location=operation.location[0])
            elif operation.num_qudits == 2:
                if ((operation.location == (1, 2) or operation.location == (2, 1))
                        and state is MachineSchedulingState.EVEN):
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    new_op = Operation(gate=operation.gate, location=(0, 1))
                    tmp_circuit.append(new_op)
                    if tmp_circuit.num_qudits == 4:
                        new_op = Operation(gate=operation.gate, location=(2, 3))
                        tmp_circuit.append(new_op)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                elif ((operation.location == (0, 1) or operation.location == (1, 0)
                       or operation.location == (2, 3) or operation.location == (3, 2))
                      and state is MachineSchedulingState.ODD):
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    new_op = Operation(gate=operation.gate, location=(1, 2))
                    tmp_circuit.append(new_op)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                else:
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    tmp_circuit.append_gate(gate=RZZGate(), location=operation.location)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
            else:
                raise ValueError("Unsupported circuit gate with big qudits")
    return tmp_circuit


class ReplacementPass(BasePass):
    """Replacement pass to choose the potential point from zone scheduler and try replacing """

    def __init__(self,
                 block_num_qudits: int = 4,
                 replacement_type: str = 'replacement',
                 qsearch_maxlayer: int = 5,
                 qsearch_heuristic_factor: int = 2
                 ) -> None:
        """
        Construct the Replacement pass

        Args:
            replacement_type (str): Type of replacement either Qsearch or intanstiation
        """

        self.block_num_qudits = block_num_qudits
        self.replacement_type = replacement_type
        self.qsearch_maxlayer = qsearch_maxlayer
        self.qsearch_heuristic_factor = qsearch_heuristic_factor
    async def run(self,
                  circuit: Circuit,
                  data: PassData) -> None:

        if (ZoneSchedulerPass.key_zones not in data or
                ZoneSchedulerPass.key_zone_weight not in data or
                ZoneSchedulerPass.key_zone_states not in data):
            raise RuntimeError(
                'Cannot find bounded shifted zone, try running a'
                ' ZoneSchedulerPass first.',
            )

        """ Problematic points are identified """
        zones = data[ZoneSchedulerPass.key_zones]
        zone_weights = data[ZoneSchedulerPass.key_zone_weight]
        zone_states = data[ZoneSchedulerPass.key_zone_states]
        potential_shift_locations = []
        potential_shift_state = []
        for zone_idx in range(1, len(zones)):
            if zone_weights[zone_idx] == 1:  # possible problematic points (Experiments needed)
                for point in zones[zone_idx]:
                    if circuit[point].gate == RZZGate():
                        potential_shift_locations.append(point)
                        potential_shift_state.append(zone_states[zone_idx])

        """ Automatic identify and re-instantiate trouble points """
        reversed_problem_points = potential_shift_locations[::-1]
        reversed_states = potential_shift_state[::-1]
        for p, q in zip(reversed_problem_points, reversed_states):
            if not p:
                continue
            _logger.debug(f"Point: {p}")
            _logger.debug(f"Machine State: {q}")
            circuit_region = circuit.surround(point=p, num_qudits=self.block_num_qudits, fail_quickly=True)
            _logger.debug(f"Circuit region: {circuit_region}")
            folded_point = circuit.fold(circuit_region)
            op = circuit.get_operation(folded_point)
            target_unitary = op.get_unitary()
            old_block_circuit = op.gate._circuit  # type: ignore
            if self.replacement_type == "instantiation":
                """ Instantiation """
                _logger.debug("Running instantiation......")
                tmp_circuit = alternate_circuit_structure(old_block_circuit, q)
                replaced_circuit = tmp_circuit.instantiate(target=target_unitary, multistarts=5)
            elif self.replacement_type == "qsearch":
                """ Qsearch """
                _logger.debug("Running Qsearch......")
                qsearch_shift_pass = QSearchSynthesisPass(
                    layer_generator=ShuttlingShiftGenerator(q),
                    max_layer=self.qsearch_maxlayer,
                    heuristic_function=HeuristicSearch(
                        heuristic_factor=self.qsearch_heuristic_factor,
                        qtm_machine=QtmMachine.H1
                    ),
                )
                sub_workflow = [qsearch_shift_pass]
                with Compiler() as compiler:
                    replaced_circuit = compiler.compile(old_block_circuit, sub_workflow)
            else:
                raise RuntimeError("The replacement type should only be instantiation or qsearch.")
            distance = replaced_circuit.get_unitary().get_distance_from(target_unitary, 2)
            print("Distance between instantiation and target unitary", distance)
            if distance < 1e-8:
                circuit.replace_with_circuit(folded_point, replaced_circuit)
                print("Successfully replace the problem point with instantiation")
            circuit.unfold_all()
            return None
