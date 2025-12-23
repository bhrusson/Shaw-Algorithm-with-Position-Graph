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
from bqskit.qis.graph import CouplingGraph

from bqskit_local.position.graph import EdgeCapability
from ..position.state import PositionGraphState



logging.basicConfig(level=logging.WARNING)
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
        F = set(circuit.front)
        decay = [1.0 for _ in range(pgs.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int,int]] = []

        _logger.debug(f'Starting forward sabre pass with pgs: {pgs}.')

        radix = circuit.radixes[0]
        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        D = pgs.position_graph.shortest_path_lengths



        while len(F) > 0:
            # Get executable gates
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            if execute_list:
                # Execute gates
                leading_swaps.clear()
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n, None)
                    _logger.debug(f'Executing gate at point {n}.')
                    
                    if modify_circuit:
                        op = circuit[n]
                        phys = [int(pgs.logical_to_physical[q]) for q in op.location]
                        mapped_circuit.append_gate(op.gate, tuple(phys), op.params)

                    # Update successors
                    for succ in circuit.next(n):
                        prev_executed_counts[succ] = prev_executed_counts.get(succ, 0) + 1
                        if prev_executed_counts[succ] == len(circuit.prev(succ)):
                            F.add(succ)

                # Reset decay
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue

            # If no gates executable, pick a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, pgs, decay)

    

            # Apply swap
            self._apply_swap(best_swap, pgs, decay)
            leading_swaps.append(best_swap)

            if modify_circuit:
                pos0, pos1 = best_swap
                q0 = pgs.get_logical_qudit_at_position(pos0)
                q1 = pgs.get_logical_qudit_at_position(pos1)

                if (q0 != -1 and q1 != -1):
                    # Both qudits exist → swap
                    _logger.warning(f"swap phys=({pos0},{pos1}) logical=({q0},{q1})")
                    _logger.warning("\physical_to_logical:" + str(pgs.physical_to_logical))
                    _logger.warning("\logical_to_physical:" + str(pgs.logical_to_physical))

                    if (q0 <= mapped_circuit.num_qudits - 1):
                        if (q1 <= mapped_circuit.num_qudits - 1):
                            mapped_circuit.append_gate(SwapGate(radix), (q0 , q1 ))
                        else:
                            mapped_circuit.append_gate(BarrierPlaceholder(1), (q0,))
                    elif (q1 <= mapped_circuit.num_qudits - 1):
                        mapped_circuit.append_gate(BarrierPlaceholder(1), (q1,))
                elif (q0 != -1) and (q1 <= mapped_circuit.num_qudits - 1) and (q1>=0):
                    mapped_circuit.append_gate(BarrierPlaceholder(1), (q1,))
                elif (q1 != -1)  and (q0 <= mapped_circuit.num_qudits - 1) and (q0>=0):
                    mapped_circuit.append_gate(BarrierPlaceholder(1), (q0,))

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

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.
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

        # Here I was thinking maybe it would be cool to support the CouplingGraph
        # and the PositionGraph as possible inputs. Maybe on a refactor
        # if that seems worthwhile.

        #physical_qudits = [pi[i] for i in op.location]
        #return cg.get_subgraph(physical_qudits).is_fully_connected()

        positions = set([pgs.logical_to_physical[i] for i in op.location])
        is_in_cluster = pgs.position_graph.in_cluster(positions)
        _logger.debug(
            "op.location: " + str(op.location) 
            + "\n positions" + str(positions) 
            + "\n is_in_cluster = pgs.position_graph.in_cluster(positions) :" + str(is_in_cluster) 
            + "\n pgs.position_graph._executable_clusters: " + str(pgs.position_graph._executable_clusters))
        return pgs.position_graph.in_cluster(positions)
            
        

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

    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[float]],
        #cg: CouplingGraph,
        #pi: list[int],
        pgs: PositionGraphState,
        decay: list[float],
    ) -> tuple[int, int]:
        """Return the best swap given the current algorithm state."""
        # Track best one
        best_score = np.inf
        best_swap = None

        # Gather all considerable swaps
        swap_candidate_list = self._obtain_swaps(circuit, F, pgs)
          # Score them, tracking the best one
        for swap in swap_candidate_list:
            score = self._score_swap(circuit, F, pgs, D, swap, decay, E)
            if score < best_score:
                best_score = score
                best_swap = swap

        if best_swap is None:
            raise RuntimeError('Unable to find best swap.')
            #_logger.error("Unable to find best swap.")
            #best_swap = random.choice(list(swap_candidate_list))
        return best_swap

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

        physical_qudits = [pgs.logical_to_physical[i] for i in qudits_in_F]
        swaps = set()

        for physical_qudit in physical_qudits:
            neighbors = pgs.position_graph.graph.neighbors_undirected(physical_qudit)
            for neighbor in neighbors:
                # check edge label for MOVE or SWAP capability
                edge_label = pgs.position_graph.edge_labels.get((physical_qudit, neighbor))
                if edge_label is None:
                    continue  # no edge here, skip
                if not (edge_label.has_capability(EdgeCapability.MOVE) or edge_label.has_capability(EdgeCapability.SWAP)):
                    continue  # edge does not allow move or swap, skip

                # valid swap, add sorted tuple to avoid duplicates
                a, b = sorted((physical_qudit, neighbor))
                swaps.add((a, b))

        _logger.warning(f"swaps for frontier F: {swaps}")
        return swaps


    def _score_swap(
        self,
        circuit,
        F,
        pgs: PositionGraphState,
        D,  # could be AllPairsPathMapping or your distance matrix
        swap: tuple[int, int],
        decay: list[float],
        extended_set
    ) -> float:
        """
        Score a candidate swap between two positions.
        """
        pos0, pos1 = swap
        q0 = pgs.get_logical_qudit_at_position(pos0)
        q1 = pgs.get_logical_qudit_at_position(pos1)

        decay_factor = 0.0
        if q0 != -1:
            decay_factor = max(decay_factor, decay[q0])
        if q1 != -1:
            decay_factor = max(decay_factor, decay[q1])

        # Compute distance heuristic for gates in F
        front = 0.0
        for cp in F:
            op = circuit[cp]
            logical_qudits = op.location
            # Only consider logical qudits that are currently assigned
            assigned = [q for q in logical_qudits if pgs.logical_to_physical[q] != -1]
            if not assigned:
                continue

            # Map logical qudits to physical positions
            phys_positions = [pgs.logical_to_physical[q] for q in assigned]

            total = 0.0
            for i, u in enumerate(phys_positions):
                for j, v in enumerate(phys_positions):
                    if i == j:
                        continue
                    try:
                        dist = D[u][v]
                    except KeyError:
                        dist = float('inf')
                    total += dist
            front += total



        # Combine distance with decay
        score = front / ( decay_factor + 1e-8)
        _logger.warning("\nswap: " + str(swap) + "\nscore:" + str(score))
        return score


    def _apply_swap(self, swap: tuple[int, int], pgs: PositionGraphState, decay: list[float]):
        """
        Apply a swap between two physical positions in the PositionGraphState.
        Updates decay for the involved logical qudits.
        """
        pos0, pos1 = swap

        # Map physical positions to logical qudits
        q0 = pgs.get_logical_qudit_at_position(pos0)
        q1 = pgs.get_logical_qudit_at_position(pos1)

        # Update decay for logical qudits only if they exist
        if q0 != -1:
            decay[q0] += self.decay_delta
        if q1 != -1:
            decay[q1] += self.decay_delta

        # Swap logical qudits in PGS if both are assigned
        if (q0 != -1 and q1 != -1):
            pgs.swap_positions(q0, q1)
        elif q0 != -1:
            # Move q0 to pos1
            pgs.move_qudit(q0, pos1)
        elif q1 != -1:
            # Move q1 to pos0
            pgs.move_qudit(q1, pos0)
        # If both are -1, nothing to do



    def _get_distance(self, logical_qudits: Sequence[int], pgs: PositionGraphState, D) -> float:
        positions = [pgs.logical_to_physical[q] for q in logical_qudits]
        total = 0.0
        for i, u in enumerate(positions):
            for j, v in enumerate(positions):
                if i == j:
                    continue
                try:
                    dist = D[u][v]
                except KeyError:
                    dist = float('inf')
                total += dist
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
                D[pgs.logical_to_physical[q]][pgs.logical_to_physical[p]]
                for p in logical_qudits if p != q
            ),
        )

        center_pos = pgs.logical_to_physical[center_qudit]


        for q in logical_qudits:
            if q == center_qudit:
                continue

            spt = pgs.get_shortest_path_tree(center_qudit)
            # Get path from this qudit to the center
            path = list(reversed(spt[pgs.logical_to_physical[q]]))

            # Yield swaps along the path to move qudit q toward the center
            for p1, p2 in zip(path, path[1:]):
                # Skip invalid qubits
                if not (0 <= p1 < pgs.num_qudits and 0 <= p2 < pgs.num_qudits):
                    continue
                # Skip if either end is already the center qudit
                if pgs.logical_to_physical[center_qudit] in (p1, p2):
                    continue
                yield (p1, p2)

    def _apply_perm(self, perm: Sequence[int], pgs: PositionGraphState) -> None:
        """Apply the `perm` permutation to the current mapping `pgs`."""
        _logger.debug('applying permutation %s' % str(perm))
        pgs_c = {q: pgs.logical_to_physical[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pgs.set_qudit_position(q,pgs_c[q])

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
        pos1 = pgs.logical_to_physical[q1]
        pos2 = pgs.logical_to_physical[q2]
        return pgs.position_graph.shortest_path(pos1, pos2)