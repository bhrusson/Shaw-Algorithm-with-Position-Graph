"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
import logging
import random
from typing import Iterator
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
#from bqskit.qis.graph import CouplingGraph

from bqskit_local.position.graph import EdgeCapability
from bqskit_local.position.state import PositionGraphState



logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)



class GeneralizedSabreAlgorithmPGS():
    """
    Implements methods for Sabre-based layout and routing algorithms using a
    modified heuristic to accommodate larger than 2-qudit gates.

    References:
        Gushu Li, Yufei Ding, and Yuan Xie. 2019. Tackling the Qubit
        Mapping Problem for NISQ-Era Quantum Devices. In Proceedings of
        the 24th ACM International Conference on Architectural
        Support for Programming Languages and Operating Systems
        (ASPLOS 2019). Association for Computing Machinery, New York, NY,
        USA, 1001-1014. https://doi.org/10.1145/3297858.3304023

        Casey Duckering, Jonathan M. Baker, Andrew Litteken, and Frederic
        T. Chong. 2021. Orchestrated trios: compiling for efficient
        communication in Quantum programs with 3-Qubit gates. In Proceedings
        of the 26th ACM International Conference on Architectural Support
        for Programming Languages and Operating Systems (ASPLOS 2021).
        Association for Computing Machinery, New York, NY, USA, 375-385.
        https://doi.org/10.1145/3445814.3446718
    """

    def __init__(
        self,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        
    ) -> None:
        """
        Construct a GeneralizedSabreAlgorithm.

        Args:
            decay_delta (float): The amount to adjust the decay factor by
                each time a swap is applied. Set to zero to disable decay.
                (Default: 0.001)

            decay_reset_interval (int): The amount of swaps to apply before
                reseting the decay factors. (Default: 5)

            decay_reset_on_gate (bool): If true, reset decay factors when
                a logical gate is applied. (Default: True)

            extended_set_size (int): The size of the look-ahead or extended
                set. Set to zero to disable look ahead. (Default: 20)

            extended_set_weight (float): The weight on the extended set
                term when scoring potential swaps. (Default: 0.5)
        """
        if not isinstance(decay_delta, float):
            raise TypeError(
                'Expected float for decay_delta'
                f', got {type(decay_delta)}',
            )

        if not isinstance(decay_reset_interval, int):
            raise TypeError(
                'Expected int for decay_reset_interval'
                f', got {type(decay_reset_interval)}',
            )

        if not isinstance(decay_reset_on_gate, bool):
            raise TypeError(
                'Expected bool for decay_reset_on_gate'
                f', got {type(decay_reset_on_gate)}',
            )

        if not isinstance(extended_set_size, int):
            raise TypeError(
                'Expected int for extended_set_size'
                f', got {type(extended_set_size)}',
            )

        if not isinstance(extended_set_weight, float):
            raise TypeError(
                'Expected float for extended_set_weight'
                f', got {type(extended_set_weight)}',
            )

        if decay_reset_interval < 1:
            raise ValueError('Decay reset interval must be a positive integer.')

        if extended_set_size < 0:
            raise ValueError('Extended set size must be a nonnegative integer.')

        self.decay_delta = decay_delta
        self.decay_reset_interval = decay_reset_interval
        self.decay_reset_on_gate = decay_reset_on_gate
        self.extended_set_size = extended_set_size
        self.extended_set_weight = extended_set_weight

    def forward_pass(
        self,
        circuit: Circuit,        
        pgs: PositionGraphState,
        modify_circuit: bool = False,
    ) -> None:
        """
        Forward pass of the Sabre algorithm 
        """
        D = pgs.position_graph.shortest_path_lengths
        F = set(circuit.front)
        decay = [1.0 for i in range(pgs.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int,int]] = []

        _logger.debug(f'Starting forward sabre pass with pgs: {pgs}.')

        radix = circuit.radixes[0]
        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        while len(F) > 0:
            # Get executable gates
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            if len(execute_list) > 0:
                # Execute gates
                leading_swaps = []
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
                    
                    if modify_circuit:
                        op = circuit[n]
                        op_positions = [int(pgs.logical_to_position[q]) for q in op.location]
                        mapped_circuit.append_gate(op.gate, op_positions, op.params)

                    # Update successors
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                # Reset decay
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue

            elif len(leading_swaps) > 5 *pgs.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)
                    if modify_circuit:
                        point = mapped_circuit._rear[swap[0]]
                        mapped_circuit.pop(point)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pgs, D),
                )
                for swap in self._uphill_swaps(qudits, pgs, D):
                    self._apply_swap(swap, pgs, decay)
                    if modify_circuit:
                        mapped_circuit.append_gate(SwapGate(radix), swap)
                _logger.debug('Stopping override.')
                continue

            # If no gates executable, pick a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, pgs, decay)

            # Apply swap
            self._apply_swap(best_swap, pgs, decay)
            leading_swaps.append(best_swap)

            if modify_circuit: 
                mapped_circuit.append_gate(SwapGate(radix), best_swap)


            # Update counters
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if modify_circuit:
            circuit.become(mapped_circuit)

    def backward_pass(
        self,
        circuit: Circuit,        
        pgs: PositionGraphState
        #pi: list[int],
        #cg: CouplingGraph        
    ) -> None:
        """
        Apply a backward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pgs : Position Graph State 
        """
        # Preprocessing
        D = pgs.position_graph.shortest_path_lengths #Might not need this here.
        F = circuit.rear
        decay = [1.0 for i in range(pgs.num_qudits)]
        iter_count = 0
        leading_swaps: list[tuple[int, int]] = []
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting backward sabre pass with pgs: {pgs}.')

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping: pi
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                leading_swaps = []

                for n in execute_list:
                    F.remove(n)
                    next_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')

                    for predessor in circuit.prev(n):
                        if predessor not in next_executed_counts:
                            next_executed_counts[predessor] = 1
                        else:
                            next_executed_counts[predessor] += 1
                        num_next_executed = next_executed_counts[predessor]
                        total_num_next = len(circuit.next(predessor))
                        if num_next_executed == total_num_next:
                            F.add(predessor)

                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue  # Restart main loop if we executed at least one gate

            # If execute list is empty, check for local-minima
            elif len(leading_swaps) > 5 * pgs.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pgs, D),
                )
                for swap in self._uphill_swaps(qudits, pgs, D):
                    self._apply_swap(swap, pgs, decay)
                _logger.debug('Stopping override.')
                continue

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, pgs, decay)     

            self._apply_swap(best_swap, pgs, decay)
            leading_swaps.append(best_swap)

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

    def _can_exe(self, op: Operation, pgs: PositionGraphState) -> bool:
        """Return true if `op` is executable given the current pgs."""
        if isinstance(op.gate, BarrierPlaceholder):
            _logger.debug("op is executable given the current pgs")
            return True

        if isinstance(op.gate, CircuitGate):
            if all(g.num_qudits == 1 for g in op.gate._circuit.gate_set):
                _logger.debug("true - isinstance(op.gate, CircuitGate): - true, - if all(g.num_qudits == 1 for g in op.gate._circuit.gate_set):")
                return True

        if op.num_qudits == 1:
            _logger.debug("1 qudit = true")
            return True
        p, q = (int(pgs.logical_to_position[i]) for i in op.location)

        exec_graph = pgs.position_graph.get_projected_graph(EdgeCapability.EXECUTE)
        return exec_graph.has_edge(p, q)

        #op_positions = set([pgs.logical_to_position[i] for i in op.location])
        #is_in_cluster = pgs.position_graph.in_cluster(op_positions)
        #_logger.debug(
        #    "op.location: " + str(op.location) 
        #    + "\n positions" + str(op_positions) 
        #    + "\n is_in_cluster = pgs.position_graph.in_cluster(positions) :" + str(is_in_cluster) 
        #    + "\n pgs.position_graph._executable_clusters: " + str(pgs.position_graph._executable_clusters))
        #return pgs.position_graph.in_cluster(op_positions)
            
        

    def _calc_extended_set(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
    ) -> set[CircuitPoint]:
        """Calculate the Extended Set for look-ahead capabilities."""
        extended_set: set[CircuitPoint] = set()
        frontier = list(copy.copy(F))
        while len(frontier) > 0 and len(extended_set) < self.extended_set_size:
            n = frontier.pop(0)
            extended_set.update(circuit.next(n))
            frontier.extend(circuit.next(n))
        return extended_set

    def _obtain_swaps(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionGraphState,
    ) -> set[tuple[int, int]]:
        """Produce all physical swaps with at least one qudit in F."""
        
        qudits_in_F: set[int] = set()
        for n in F:
            qudits_in_F.update(circuit[n].location)

        # Map logical qudits to physical positions
        F_positions = [
            int(pgs.logical_to_position[q]) 
            for q in qudits_in_F 
            if pgs.logical_to_position[q] != -1
        ]

        swaps: set[tuple[int, int]] = set()

        for pos in F_positions:
            neighbors = pgs.position_graph.graph.neighbors_undirected(pos)
            for neighbor in neighbors:
                neighbor = int(neighbor)

                edge_label = pgs.position_graph.edge_labels.get((pos, neighbor))
                if edge_label is None:
                    continue
                if not (edge_label.has_capability(EdgeCapability.MOVE) or
                        edge_label.has_capability(EdgeCapability.SWAP)):
                    continue

                a, b = sorted((pos, neighbor))
                swaps.add((a, b))

        if not swaps:
            _logger.warning(
                f"No swaps found for frontier positions {F_positions}. "
                f"Frontier qudits: {qudits_in_F}"
            )

        return swaps


    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[float]],
        pgs: PositionGraphState,
        decay: list[float],
    ) -> tuple[int, int]:
        """Return the best swap given the current algorithm state."""
        
        swap_candidate_list = self._obtain_swaps(circuit, F, pgs)
        if not swap_candidate_list:
            raise RuntimeError("No valid swaps available for frontier F.")

        best_score = float('inf')
        best_swap = None

        for swap in swap_candidate_list:
            score = self._score_swap(circuit, F, pgs, D, swap, decay, E)
            if score < best_score:
                best_score = score
                best_swap = swap

        # Fallback: pick a random swap if all swaps are bad
        if best_swap is None:
            _logger.warning("All candidate swaps scored inf, picking random swap.")
            best_swap = random.choice(list(swap_candidate_list))

        return best_swap

    def _score_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionGraph,
        D: list[list[float]],
        swap: Tuple[int, int],
        decay: Optional[list[float]] = None,
        E: Optional[set[CircuitPoint]] = None
    ) -> float:
        """Optimized swap scoring for 1D line graph using PositionGraph."""
        if E is None:
            E = set()

        l1, l2 = swap

        # Make copies of logical<->position mappings
        temp_logical_to_pos = pgs.logical_to_position.copy()
        temp_pos_to_logical = pgs.position_to_logical.copy()

        # Apply swap
        pos1 = temp_logical_to_pos[l1]
        pos2 = temp_logical_to_pos[l2]
        temp_logical_to_pos[l1] = pos2
        temp_logical_to_pos[l2] = pos1
        temp_pos_to_logical[pos1] = l2
        temp_pos_to_logical[pos2] = l1

        total = 0.0

        # Score the frontier set
        for cp in F:
            logical_qudits = circuit[cp].location
            positions = [temp_logical_to_pos[q] for q in logical_qudits]
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i + 1:]:
                    total += abs(pos1 - pos2)

        # Score the extended set
        for cp in E:
            logical_qudits = circuit[cp].location
            positions = [temp_logical_to_pos[q] for q in logical_qudits]
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i + 1:]:
                    total += abs(pos1 - pos2) * getattr(self, "extended_set_weight", 1.0)

        # Apply decay if provided
        if decay is not None:
            total += 2.0 * (decay[l1] + decay[l2])

        return total

    def _get_distance(
            self, 
            logical_qudits: Sequence[int], 
            pgs: PositionGraphState, 
            D: list[list[float]]
            ) -> float:
        positions = [pgs.logical_to_position[q] for q in logical_qudits]
        total = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total += pgs.position_graph.distance(positions[i], positions[j])
        return total

    def _uphill_swaps(
        self,
        logical_qudits: Sequence[int],
        pgs: PositionGraphState,
        #cg: CouplingGraph,
        #pi: list[int],
        D: list[list[float]],
    ) -> Iterator[tuple[int, int]]:
        """Yield the swaps necessary to bring some of the qudits together."""

        center_qudit = min(
            logical_qudits,
            key=lambda q: sum(
                D[pgs.logical_to_position[q]][pgs.logical_to_position[p]]
                for p in logical_qudits if p != q
            ),
        )

        center_pos = pgs.logical_to_position[center_qudit]


        for q in logical_qudits:
            if q == center_qudit:
                continue

            spt = pgs.position_graph.get_shortest_path_tree(center_pos)
            # Get path from this qudit to the center
            path = list(reversed(spt[pgs.logical_to_position[q]]))

            # Yield swaps along the path to move qudit q toward the center
            for p1, p2 in zip(path, path[1:]):
                if 0 <= p1 < pgs.num_qudits and 0 <= p2 < pgs.num_qudits:
                    yield (p1, p2)

    def compute_perm(self, circuit: Circuit, pgs: PositionGraphState, placement):
        perm = list(range(circuit.num_qudits))

        for _ in range(self.total_passes):
            self.forward_pass(circuit, pgs, False)
            self.backward_pass(circuit, pgs)

        return perm

    def _apply_perm(self, perm: list[int], pgs: PositionGraphState) -> None:
        """
        Apply a full permutation of logical qudits to positions safely.
        """
        # Keep track of which positions are already free
        temp_pos_to_logical = pgs.position_to_logical[:]
        temp_logical_to_pos = pgs.logical_to_position[:]

        for logical, target_pos in enumerate(perm):
            current_pos = temp_logical_to_pos[logical]
            if current_pos == target_pos:
                continue  # already in place

            # Swap logical qudits at current_pos and target_pos
            other_logical = temp_pos_to_logical[target_pos]

            # Update temp state
            temp_pos_to_logical[current_pos], temp_pos_to_logical[target_pos] = other_logical, logical
            temp_logical_to_pos[logical] = target_pos
            if other_logical != -1:
                temp_logical_to_pos[other_logical] = current_pos

        # Now commit to real PGS
        for logical, pos in enumerate(temp_logical_to_pos):
            pgs.set_qudit_position(logical, pos)


    def _apply_swap(
        self,
        swap: tuple[int, int],
        pgs: PositionGraphState,
        decay: list[float] | None = None
    ) -> None:
        """Apply a physical swap to the PGS state, optionally updating decay."""
        i, j = swap
        l_i = pgs.position_to_logical[i]
        l_j = pgs.position_to_logical[j]

        # Swap logical positions
        pgs.position_to_logical[i] = l_j
        pgs.position_to_logical[j] = l_i

        if l_i != -1:
            pgs.logical_to_position[l_i] = j
        if l_j != -1:
            pgs.logical_to_position[l_j] = i

        # Update decay if provided
        if decay is not None:
            if l_i != -1:
                decay[l_i] += 1
            if l_j != -1:
                decay[l_j] += 1

        _logger.debug(f"applied swap {swap}")



    def _pg_distance(
        self,
        q1: int,
        q2: int,
        pgs: PositionGraphState
    ) -> float:
        """
        Return the shortest-path distance between two logical qudits
        on the current PositionGraph.
        """
        pos1 = pgs.logical_to_position[q1]
        pos2 = pgs.logical_to_position[q2]
        return pgs.position_graph.shortest_path(pos1, pos2)