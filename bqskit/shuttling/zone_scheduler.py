"""This module implements the Zone Scheduling pass."""
from __future__ import annotations
from enum import Enum

import logging
import numpy as np
from bqskit import Circuit
from bqskit.ir import Operation
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData


class MachineSchedulingState(Enum):
    """Shift State of a Shuttling Machine."""
    EVEN = 0
    ODD = 1

    def flip(self) -> MachineSchedulingState:
        """Flip the state."""
        if self is MachineSchedulingState.EVEN:
            return MachineSchedulingState.ODD
        else:
            return MachineSchedulingState.EVEN


def matches_state(op: Operation, state: MachineSchedulingState) -> bool:
    """Check if the operation matches the state."""
    if op.num_qudits == 1:
        return True

    min_operation = np.min(op.location)
    return (
            (state == MachineSchedulingState.EVEN and min_operation % 2 == 0)
            or
            (state == MachineSchedulingState.ODD and min_operation % 2 == 1)
    )


# Zone is overloaded with GateZone
# ShiftBoundedZone = list[CircuitPoint]
_logger = logging.getLogger(__name__)


class ZoneSchedulerPass(BasePass):
    """ZoneScheduler pass to partition circuit into zones bounded by shift operations """
    key_zones = "__ZoneScheduler_zones"
    key_zone_weight = "__ZoneScheduler_zone_weight"
    key_zone_states = "__ZoneScheduler_zone_states"

    def __init__(self,
                 machine_state: MachineSchedulingState = MachineSchedulingState.EVEN
                 ) -> None:
        """
        Construct the ZoneSchedulerPass.

        Args:
            machine_state (MachineSchedulingState): The initial state of the machine
        """

        self.machine_state = machine_state

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
         Separate the circuit into zones delimited by shifts.

         The machine will always start in the even state.
         """
        zones = []
        machine_states = []
        zones_weight = []
        frontier = circuit.front
        to_be_processed = dict()
        while len(frontier) != 0:
            _logger.debug(f"Current machine state: {self.machine_state}")
            zone = []
            weight = 0
            investigating_further = True
            # Select the executable points wrt the current machine states
            while investigating_further:
                processed_points = []
                investigating_further = False
                next_layers_gate = set()

                for location in frontier:
                    processed = False
                    op = circuit.get_operation(location)
                    if matches_state(op, self.machine_state):
                        _logger.debug(f"{op} is added to current zone.")
                        zone.append(location)
                        if op.num_qudits == 2:  # can restrict to certain gate types
                            weight += 1
                        processed_points.append(location)
                        processed = True

                    if len(circuit.next(location)) != 0 and processed is True:
                        for new_location in circuit.next(location):
                            if new_location in to_be_processed:
                                to_be_processed[new_location] += 1
                            else:
                                to_be_processed[new_location] = 1
                            if to_be_processed[new_location] == len(circuit.prev(new_location)):
                                investigating_further = True
                                # _logger.debug(f"Gate {circuit.get_operation(new_location)} is added inside iteration")
                                next_layers_gate.add(new_location)

                # Modifying the frontier
                for location in processed_points:
                    frontier.remove(location)
                for new_location in next_layers_gate:
                    frontier.add(new_location)

            # _logger.debug(f"Frontier at the next zone: {frontier}")
            zones.append(zone)
            zones_weight.append(weight)
            machine_states.append(self.machine_state)
            self.machine_state = self.machine_state.flip()

        _logger.debug(f"Number of zones: {len(zones)}")
        gates_per_zone = []
        for zone in zones:
            gates_per_zone.append(len(zone))
        _logger.debug(f"Number of gates in all zones: {np.sum(gates_per_zone)}")
        data[self.key_zones] = zones
        data[self.key_zone_states] = machine_states
        data[self.key_zone_weight] = zones_weight
        return None
