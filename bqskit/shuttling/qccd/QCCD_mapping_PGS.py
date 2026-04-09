"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations
import math
import copy
import logging
import itertools
import os
from collections.abc import Callable
from typing import Iterator
from typing import Sequence
from itertools import permutations, combinations
import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.position_graph_state_PGS import PositionAssignmentTracker
from bqskit.shuttling.qccd.position_graph_state_PGS import PositionGraphState
from bqskit.ir.gates.barrier import BarrierPlaceholder

_logger = logging.getLogger(__name__)


PositionState = PositionGraphState | PositionAssignmentTracker


class QCCDMappingAlgorithm:
    """
    Implements methods for Sabre-based QCCD layout and routing algorithms using a
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

        J. Liu, E. Younis, M. Weiden, P. Hovland, J. Kubiatowicz and C. Iancu,
        "Tackling the Qubit Mapping Problem with Permutation-Aware Synthesis,"
        2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
         Bellevue, WA, USA, 2023, pp. 745-756,  https://doi.org/10.1109/QCE57702.2023.00090.

    """

    def __init__(
            self,
            decay_delta: float = 0.001,
            decay_reset_interval: int = 5,
            decay_reset_on_gate: bool = True,
            extended_set_size: int = 5,
            extended_set_weight: float = 0.5,
            qccd_machine: QCCDMachineModel = None,
            cogestion_rate: float = 0.6,
            force_bruteforce: bool = False,
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

            qccd_machine (QCCDMachineModel): Machine model of current QCCD hardware
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
        self.qccd_machine = qccd_machine
        self.cogestion_segment_rate = cogestion_rate
        self.force_bruteforce = force_bruteforce
        self._congestion_geometry_cache: dict[
            tuple[int, int, int, int],
            tuple[tuple[int, ...], ...],
        ] = {}

    def _make_pgs(self, ion_assignment: dict[int, int]) -> PositionGraphState:
        return self.qccd_machine.build_pgs_from_assignment(ion_assignment)

    def _full_assignment_from_pgs(self, pgs: PositionGraphState) -> dict[int, int]:
        return {
            int(logical): int(position)
            for logical, position in enumerate(pgs.logical_to_position)
            if int(position) != -1
        }

    def _program_assignment_from_pgs(
        self,
        pgs: PositionGraphState,
        program_ion_ids: Sequence[int],
    ) -> dict[int, int]:
        assignment: dict[int, int] = {}
        for logical in program_ion_ids:
            position = int(pgs.logical_to_position[int(logical)])
            if position == -1:
                raise RuntimeError(f'Program ion {logical} is not placed.')
            assignment[int(logical)] = position
        return assignment

    def _assignment_from_pgs(self, pgs: PositionGraphState) -> dict[int, int]:
        # Legacy helper kept for compatibility with older debug and compare
        # paths. This returns every placed ion tracked in the PGS state.
        return self._full_assignment_from_pgs(pgs)

    def _scratch_pgs(
        self,
        pgs: PositionState,
        slot: str,
    ) -> PositionAssignmentTracker:
        scratch_states = getattr(self, '_scratch_pgs_states', None)
        if scratch_states is None:
            scratch_states = {}
            self._scratch_pgs_states = scratch_states
        scratch = scratch_states.get(slot)
        if scratch is None:
            scratch = PositionAssignmentTracker(
                len(pgs.logical_to_position),
                len(pgs.position_to_logical),
            )
            scratch_states[slot] = scratch
        return scratch.load_from_state(pgs)

    def _log_pgs(
        self,
        pgs: PositionState,
        *,
        slot: str | None = None,
    ) -> PositionAssignmentTracker:
        if slot is not None:
            return self._scratch_pgs(pgs, slot)
        return PositionAssignmentTracker(
            len(pgs.logical_to_position),
            len(pgs.position_to_logical),
        ).load_from_state(pgs)

    def _logical_at_position(self, pos: int, pgs: PositionState) -> int | None:
        logical = int(pgs.get_logical_qudit_at_position(pos))
        return None if logical == -1 else logical

    def _is_occupied(self, pos: int, pgs: PositionState) -> bool:
        return self._logical_at_position(pos, pgs) is not None

    def _position_of_qudit(self, qudit: int, pgs: PositionState) -> int:
        position = int(pgs.get_position_of_qudit(qudit))
        if position == -1:
            raise RuntimeError(f'Logical qudit {qudit} is not placed.')
        return position

    def _snapshot_logical_positions(
        self,
        pgs: PositionState,
        logicals: Sequence[int],
    ) -> dict[int, int]:
        snapshot: dict[int, int] = {}
        for logical in logicals:
            logical_int = int(logical)
            if logical_int in snapshot:
                continue
            snapshot[logical_int] = self._position_of_qudit(logical_int, pgs)
        return snapshot

    def _restore_logical_positions(
        self,
        pgs: PositionState,
        snapshot: dict[int, int],
    ) -> None:
        affected_positions = {
            int(position) for position in snapshot.values() if int(position) != -1
        }
        for logical in snapshot:
            current_position = int(pgs.get_position_of_qudit(int(logical)))
            if current_position != -1:
                affected_positions.add(current_position)

        for position in affected_positions:
            pgs.position_to_logical[int(position)] = -1
        for logical in snapshot:
            pgs.logical_to_position[int(logical)] = -1
        for logical, position in snapshot.items():
            if int(position) == -1:
                continue
            pgs.logical_to_position[int(logical)] = int(position)
            pgs.position_to_logical[int(position)] = int(logical)

    def _sorted_points(self, points) -> list[CircuitPoint]:
        return sorted(points, key=lambda p: (p.cycle, p.qudit))

    def _sorted_unique_ints(self, values) -> list[int]:
        return sorted(set(int(v) for v in values))

    def _sorted_moves(self, moves) -> list[tuple[int, int]]:
        return sorted((int(a), int(b)) for a, b in moves)

    def _canonicalize_move_for_pgs(
        self,
        move: tuple[int, int],
        pgs: PositionState,
    ) -> tuple[int, int] | None:
        u, v = (int(move[0]), int(move[1]))
        logical_u = self._logical_at_position(u, pgs)
        logical_v = self._logical_at_position(v, pgs)
        if logical_u is None and logical_v is None:
            return None
        if logical_u is None:
            return (v, u)
        return (u, v)

    def _append_logged_move_instruction(
        self,
        instructions_list: list[list[str]],
        move: tuple[int, int],
        pgs: PositionState,
        D: list[list[float]],
    ) -> bool:
        canonical_move = self._canonicalize_move_for_pgs(move, pgs)
        if canonical_move is None:
            return False
        self._apply_move(canonical_move, pgs=pgs)
        instructions_list.append([
            f"Move {canonical_move}",
            f"{self._assignment_from_pgs(pgs)}",
            f"cost: {D[canonical_move[0]][canonical_move[1]]} seconds",
        ])
        return True

    def _apply_and_append_move(
        self,
        leading_moves: list[tuple[int, int]],
        move: tuple[int, int],
        pgs: PositionState,
        *,
        context: str = '',
    ) -> bool:
        canonical_move = (int(move[0]), int(move[1]))
        if canonical_move[0] == canonical_move[1]:
            message = f'Skipping no-op move {canonical_move}'
            if context:
                message += f' ({context})'
            self._debug_compare(message)
            _logger.debug(message)
            return False
        self._apply_move(canonical_move, pgs=pgs)
        leading_moves.append(tuple(sorted(canonical_move)))
        return True

    @property
    def _log_prefix(self) -> str:
        return '[PGS]'

    def _status_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_VERBOSE', '1').lower() not in (
            '0', 'false', 'no', 'off',
        )

    def _status(self, message: str | Callable[[], str]) -> None:
        if self._status_enabled():
            rendered = message() if callable(message) else message
            print(f'{self._log_prefix} {rendered}')

    def _debug_compare_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_COMPARE_DEBUG', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _debug_compare(self, message: str | Callable[[], str]) -> None:
        if self._debug_compare_enabled():
            rendered = message() if callable(message) else message
            print(f'{self._log_prefix}[compare] {rendered}')

    def _capture_forward_trace_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_CAPTURE_TRACE', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _snapshot_trace_value(self, value: object) -> object:
        """Copy simple trace payloads without paying full deepcopy overhead."""
        if isinstance(value, dict):
            return {
                key: self._snapshot_trace_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._snapshot_trace_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._snapshot_trace_value(item) for item in value)
        return value

    def _append_resolve_trace(self, entry: dict[str, object]) -> None:
        if not self._capture_forward_trace_enabled():
            return
        if not hasattr(self, 'last_bruteforce_trace') or self.last_bruteforce_trace is None:
            return
        traces = self.last_bruteforce_trace.setdefault('resolve_trace', [])
        traces.append(self._snapshot_trace_value(entry))

    def _congestion_layers(
        self,
        position: int,
        target: int,
        blockage: int,
        depth: int,
    ) -> tuple[tuple[int, ...], ...]:
        key = (int(position), int(target), int(blockage), int(depth))
        cached = self._congestion_geometry_cache.get(key)
        if cached is not None:
            return cached

        depth_d_neighbors: list[tuple[int, ...]] = []
        full_neighbors: set[int] = set()
        while len(depth_d_neighbors) < depth:
            if len(depth_d_neighbors) == 0:
                node_d_neighbors = [
                    int(node)
                    for node in self.qccd_machine.get_move_neighbors(position)
                    if int(node) != int(blockage)
                ]
                full_neighbors.update(node_d_neighbors)
            else:
                node_d_neighbors = []
                for pos in depth_d_neighbors[-1]:
                    for node in self.qccd_machine.get_move_neighbors(int(pos)):
                        node_int = int(node)
                        if (
                            node_int not in full_neighbors
                            and node_int != int(target)
                            and node_int != int(blockage)
                            and node_int != int(position)
                        ):
                            node_d_neighbors.append(node_int)
                            full_neighbors.add(node_int)
            depth_d_neighbors.append(tuple(self._sorted_unique_ints(node_d_neighbors)))

        cached = tuple(depth_d_neighbors)
        self._congestion_geometry_cache[key] = cached
        return cached

    def _congestion_signature(
        self,
        position: int,
        layers: tuple[tuple[int, ...], ...],
        pgs: PositionState,
    ) -> tuple[int, ...]:
        occupied: list[int] = []
        position_int = int(position)
        if self._is_occupied(position_int, pgs):
            occupied.append(position_int)
        for layer in layers:
            for node in layer:
                node_int = int(node)
                if self._is_occupied(node_int, pgs):
                    occupied.append(node_int)
        return tuple(occupied)

    def _congestion_rate_from_layers(
        self,
        position: int,
        layers: tuple[tuple[int, ...], ...],
        pgs: PositionState,
    ) -> tuple[float, float]:
        position_int = int(position)
        if self._is_occupied(position_int, pgs):
            congestion_score = 1.0
        else:
            congestion_score = 0.0

        layer_weight = 1.0
        num_neighbors = 0
        num_occupied_neighbors = 0
        seen_neighbors: set[int] = set()
        for layer in layers:
            for node in layer:
                node_int = int(node)
                if self._is_occupied(node_int, pgs):
                    congestion_score += layer_weight
                if node_int not in seen_neighbors:
                    seen_neighbors.add(node_int)
                    num_neighbors += 1
                    if self._is_occupied(node_int, pgs):
                        num_occupied_neighbors += 1
            layer_weight -= 0.1

        if num_neighbors == 0:
            return 1.0, np.inf
        return num_occupied_neighbors / num_neighbors, congestion_score

    def _cached_congestion_rate(
        self,
        position: int,
        target: int,
        blockage: int,
        pgs: PositionState,
        *,
        depth: int,
        cache: dict[tuple[object, ...], tuple[float, float]],
    ) -> tuple[float, float]:
        layers = self._congestion_layers(position, target, blockage, depth)
        key = (
            int(position),
            int(target),
            int(blockage),
            int(depth),
            self._congestion_signature(position, layers, pgs),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        cached = self._congestion_rate_from_layers(position, layers, pgs)
        cache[key] = cached
        return cached

    def _format_locations(
        self,
        circuit: Circuit,
        points: Sequence[CircuitPoint],
    ) -> list[tuple[int, ...]]:
        return [
            tuple(int(q) for q in circuit[n].location)
            for n in self._sorted_points(points)
        ]

    def _record_forward_trace(
        self,
        *,
        circuit: Circuit,
        front_points: Sequence[CircuitPoint],
        execute_list: Sequence[CircuitPoint],
        pgs: PositionGraphState,
        pre_assignment: dict[int, int],
        post_assignment: dict[int, int],
        action: str,
        best_move: tuple[int, int] | None = None,
        brute_force_gate: tuple[int, ...] | None = None,
    ) -> None:
        if not self._capture_forward_trace_enabled():
            return
        sorted_front = self._sorted_points(front_points)
        brute_force_trace = None
        if action == 'bruteforce':
            brute_force_trace = self._snapshot_trace_value(
                getattr(self, 'last_bruteforce_trace', None),
            )
        self.last_forward_trace.append({
            'action': action,
            'front_points': [str(n) for n in sorted_front],
            'front_locations': self._format_locations(circuit, sorted_front),
            'front_executable': [
                (
                    tuple(int(q) for q in circuit[n].location),
                    self.qccd_machine.gate_is_executable(
                        circuit[n],
                        list(range(circuit.num_qudits)),
                        pre_assignment,
                    ),
                )
                for n in sorted_front
            ],
            'execute_points': [str(n) for n in self._sorted_points(execute_list)],
            'execute_locations': self._format_locations(circuit, execute_list),
            'best_move': None if best_move is None else tuple(int(x) for x in best_move),
            'brute_force_gate': brute_force_gate,
            'brute_force_trace': brute_force_trace,
            'pre_assignment': dict(pre_assignment),
            'post_assignment': dict(post_assignment),
        })

    def forward_pass(
            self,
            circuit: Circuit,
            pgs: PositionGraphState,
            modify_circuit: bool = False,
    ) -> None:
        """
        Apply a forward pass of the QCCD mapper using PositionGraphState.

        Args:
            circuit (Circuit): The circuit to pass over.

            pgs (PositionGraphState): The live logical-to-position state.

            modify_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        # print("The position graph: ", self.qccd_machine.position_graph)
        # if not self.qccd_machine.check_valid_assignment(ion_assignment):
        #     raise ValueError("The ion assignment is not valid."
        #                      " There is either repetition in the assignment
        #                      or the ions are not initially inside traps.")
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.front
        decay = [1.0 for _ in range(self.qccd_machine.num_positions)]
        repeated_path = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        self.last_forward_trace: list[dict[str, object]] = []
        capture_forward_trace = self._capture_forward_trace_enabled()
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                '%s Starting forward sabre pass with ion assignment: %s.',
                self._log_prefix,
                self._assignment_from_pgs(pgs),
            )
        # print(f"Starting forward sabre pass with ion assignment: {ion_assignment}.")

        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(
                self.qccd_machine.num_positions,
                [circuit.radixes[0]] * self.qccd_machine.num_positions,
            )
            barrier_qudits = list(range(self.qccd_machine.num_positions))

        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')
        longest_path = self.qccd_machine.get_longest_move_path_length()
        # Main Loop
        executed_flag = False
        heuristic_move = True
        while len(F) > 0:
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                # print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path / 4):
                self._status(
                    f'Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate',
                )
                log_pgs = self._log_pgs(pgs, slot='forward_log')
                brute_force_moves = self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pgs,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_pgs,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                        )
                leading_moves += brute_force_moves
            current_front = self._sorted_points(F)
            pre_assignment = (
                self._assignment_from_pgs(pgs)
                if capture_forward_trace
                else None
            )
            execute_list = self._sorted_points([
                n for n in F
                if self.qccd_machine.gate_is_executable_pgs(circuit[n], pgs)
            ])
            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                # Rest penalty from reptition
                self.extended_set_size = initial_extended_set_size
                # Add the temporary F to current F
                if tmp_F:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug('%s Executing gate at point %s.', self._log_prefix, n)
                    self._status(f'Executing gate at point {n}.')
                    if modify_circuit:
                        op = circuit[n]
                        physical_location = [self._position_of_qudit(q, pgs) for q in op.location]
                        mapped_circuit.append_gate(op.gate, physical_location)
                        instructions_list.append(
                            [f"Execute at {physical_location}", f"{self._assignment_from_pgs(pgs)}"],
                        )
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                        )
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)
                if capture_forward_trace:
                    self._record_forward_trace(
                        circuit=circuit,
                        front_points=current_front,
                        execute_list=execute_list,
                        pgs=pgs,
                        pre_assignment=pre_assignment,
                        post_assignment=self._assignment_from_pgs(pgs),
                        action='execute',
                    )
                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    self.iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue  # Restart main loop if we executed at least one gate

            executed_flag = False
            # Pick and apply a swap
            if repeated_path:
                # If there is repetition, first take into account only one gate in F
                # Then change the extended set size to 0
                repeated_path = False
                if len(F) == 1 and self.extended_set_size != 0:
                    self.extended_set_size = 0
                elif len(F) == 1 and self.extended_set_size == 0:
                    # Retrieve executable gates giving the current ion assignment `pi`
                    if self.iter_count > 2:
                        self._status('Try bruteforce due to repeated pattern...')
                        log_pgs = self._log_pgs(pgs, slot='forward_log')
                        brute_force_moves = self._brute_force_congestion(
                            circuit[self._sorted_points(F)[0]],
                            D,
                            pgs,
                        )
                        if modify_circuit:
                            for move in brute_force_moves:
                                if not self._append_logged_move_instruction(
                                    instructions_list,
                                    move,
                                    log_pgs,
                                    D,
                                ):
                                    continue
                                mapped_circuit.append_gate(
                                    BarrierPlaceholder(self.qccd_machine.num_positions),
                                    barrier_qudits,
                                )
                        leading_moves += brute_force_moves
                        continue
                else:
                    ordered_F = self._sorted_points(F)
                    tmp_F = ordered_F[1:]
                    F = [ordered_F[0]]
                    # print(f"Front is modified to {F}.")
            E = self._calc_extended_set(circuit, F)
            # print(f"Extended set: {[circuit[n] for n in E]}")
            if self.force_bruteforce:
                best_move = None
            else:
                best_move = self._get_best_move(
                    circuit,
                    F,
                    E,
                    D,
                    decay,
                    heuristic_move,
                    pgs,
                )
            if best_move is None:
                brute_force_gate_point = self._sorted_points(F)[0]
                brute_force_gate = tuple(
                    int(q) for q in circuit[brute_force_gate_point].location
                )
                log_pgs = self._log_pgs(pgs, slot='forward_log')
                brute_force_moves = self._brute_force_congestion(
                    circuit[brute_force_gate_point],
                    D,
                    pgs,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_pgs,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                                )
                leading_moves += brute_force_moves
                if capture_forward_trace:
                    self._record_forward_trace(
                        circuit=circuit,
                        front_points=current_front,
                        execute_list=execute_list,
                        pgs=pgs,
                        pre_assignment=pre_assignment,
                        post_assignment=self._assignment_from_pgs(pgs),
                        action='bruteforce',
                        brute_force_gate=brute_force_gate,
                    )
                continue
            self._status(f'Best move: {best_move}')
            log_pgs = self._log_pgs(pgs, slot='forward_log')
            self._apply_move(best_move, pgs=pgs)
            leading_moves.append(best_move)
            if capture_forward_trace:
                self._record_forward_trace(
                    circuit=circuit,
                    front_points=current_front,
                    execute_list=execute_list,
                    pgs=pgs,
                    pre_assignment=pre_assignment,
                    post_assignment=self._assignment_from_pgs(pgs),
                    action='move',
                    best_move=best_move,
                )

            if modify_circuit:
                if self._append_logged_move_instruction(
                    instructions_list,
                    best_move,
                    log_pgs,
                    D,
                ):
                    mapped_circuit.append_gate(
                        BarrierPlaceholder(self.qccd_machine.num_positions),
                        barrier_qudits,
                    )

            self.iter_count += 1
        if modify_circuit:
            circuit.become(mapped_circuit)
            return instructions_list

    ########################Local minima resolution#############################
    def _brute_force_congestion(
            self,
            gate: Operation,
            D: list[list[float]],
            pgs: PositionGraphState,
    ) -> list[tuple[int, int]]:
        """
            Logical function
        """
        gate_pos = []
        leading_moves = []
        for p in gate.location:
            gate_pos.append(self._position_of_qudit(p, pgs))
        initial_gate_pos = list(gate_pos)
        self._status(
            lambda: (
                f'Trying to solve brute-force congestion at gate {gate_pos} '
                f'with {self._assignment_from_pgs(pgs)}'
            ),
        )
        #raise ValueError("Stopping for debug....")
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                '%s Trying to solve brute-force congestion at gate %s with %s',
                self._log_prefix,
                gate_pos,
                self._assignment_from_pgs(pgs),
            )
        selected_trap_space = []
        selected_end_point = []
        relative_distance = np.inf
        selected_trap_id = None
        # Select which trap to brute force in
        for trap in self.qccd_machine.physical_graph.executable_trap_list:
            all_trap_space = list(self.qccd_machine.physical_to_position[trap.id])
            # endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
            _, available_trap_space = self.qccd_machine.trap_is_fully_occupied_pgs(trap.id, pgs)
            # Change to endpoints of trap space ... TODO
            relative_dis_to_trap = self._get_distance_from_position_to_trap(gate_pos,
                                                                            all_trap_space,
                                                                            D,
                                                                            pgs)
            """
                Only need to calculate unoccupied spaces (the one near the endpoints) if exits or only the endpoints...  
            """
            num_available_trap_space = len(available_trap_space)
            relative_dis_to_trap -= num_available_trap_space * 120e-6
            # print(f"Considering trap: {trap.id} with distance {relative_dis_to_trap} and number of available space :{num_available_trap_space}")
            if relative_dis_to_trap < relative_distance:
                selected_trap_space = all_trap_space
                # ToDo: If there are more than two endpoints?
                # selected_end_point = self.qccd_machine.trap_end_points[trap.id]
                selected_trap_id = trap.id
                relative_distance = relative_dis_to_trap
        selected_trap_space_unsorted = list(selected_trap_space)
        # print(f"Selected trap: {selected_trap_space}", )
        # Select the order of moving position
        distance_to_trap_lst = []
        for pos in gate_pos:
            tmp_distance_to_trap = [D[pos][trap_space] for trap_space in selected_trap_space]
            distance_to_trap = float(np.min(tmp_distance_to_trap))
            end_point = selected_trap_space[int(np.argmax(tmp_distance_to_trap))]
            if end_point not in self.qccd_machine.trap_end_points[selected_trap_id]:
                end_point = self.qccd_machine.trap_end_points[selected_trap_id][0]
            selected_end_point.append(end_point)
            distance_to_trap_lst.append(distance_to_trap)
        gate_pos = np.array(gate_pos)[np.argsort(distance_to_trap_lst)]
        selected_end_point = np.array(selected_end_point)[np.argsort(distance_to_trap_lst)]
        ion_order = [self._logical_at_position(int(i), pgs) for i in gate_pos]
        _logger.debug(f"Selected end point: {selected_end_point}", )
        # print(f"Selected end point: {selected_end_point}")
        # print("Order of moving ions: ", ion_order)
        for ion_index in range(len(ion_order)):
            gate_pos[ion_index] = self._position_of_qudit(ion_order[ion_index], pgs)
        # print("Gate pos: ", gate_pos)
        # Select the trap space order
        trap_space_distance_to_end_point = []
        #for gate
        for trap_space in selected_trap_space:
            trap_space_distance_to_end_point.append(
                float(np.min([D[trap_space][end_point] for end_point in selected_end_point])))
        selected_trap_space = np.array(selected_trap_space)[np.argsort(trap_space_distance_to_end_point)]
        self.last_bruteforce_trace = {
            'gate_location': tuple(int(q) for q in gate.location),
            'initial_gate_pos': [int(x) for x in initial_gate_pos],
            'selected_trap_id': selected_trap_id,
            'selected_trap_space_initial': [int(x) for x in selected_trap_space_unsorted],
            'selected_trap_space_sorted': [int(x) for x in list(selected_trap_space)],
            'selected_end_point': [int(x) for x in list(selected_end_point)],
            'distance_to_trap_lst': [float(x) for x in distance_to_trap_lst],
            'ion_order': [int(x) for x in ion_order],
            'leading_moves': [],
        }
        congestion_cache: dict[tuple[object, ...], tuple[float, float]] = {}
        # print("Order selected traps: ", selected_trap_space)

        # Move the pos to the selected trap
        for pos_idx in range(len(gate_pos)):
            self._status(
                f'Trying to moving ion {gate_pos[pos_idx]}... to '
                f'{int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx])}',
            )
            #_logger.debug(f"Trying to moving ion {gate_pos[pos_idx]}...")
            # if pos_idx != len(gate_pos) - 1:
            #     print(f"Endpoint: {selected_end_point[pos_idx+1]}")
            leading_moves += self._brute_force_move(
                int(gate_pos[pos_idx]),
                int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx]),
                pgs,
                congestion_cache=congestion_cache,
            )
            self.last_bruteforce_trace['leading_moves'] = [
                tuple(int(v) for v in move) for move in leading_moves
            ]
            self._status(
                lambda pos_idx=pos_idx: (
                    f'Ion assignment after moving ion {gate_pos[pos_idx]}: '
                    f'{self._assignment_from_pgs(pgs)}'
                ),
            )
            # _logger.debug(f"Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}")
            # _logger.debug(f"Selected end point: {selected_end_point}")
            gate_pi = list(gate.location)
            # _logger.debug(f"Gate pi: {gate_pi}")
            # if selected_end_point in ion_assignment.values():
            #     _logger.debug(f"Position of ion: {list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point)]}")
            """
                If too many ions are in the segment, move them back to trap.
            """
            number_of_segment = len(self.qccd_machine.physical_to_position["segment_space"])
            current_assignment = self._assignment_from_pgs(pgs)
            ion_at_segment = []
            for ion in current_assignment:
                pos = self._position_of_qudit(ion, pgs)
                if pos in self.qccd_machine.physical_to_position["segment_space"]:
                    ion_at_segment.append(ion)
            if len(ion_at_segment) / number_of_segment >= self.cogestion_segment_rate:
                self._status('As there are many ions outside the traps, move them to the trap...')
                available_spaces = []
                for trap in self.qccd_machine.physical_graph.trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied_pgs(trap.id, pgs)
                    available_spaces += available_space
                while ion_at_segment:
                    leading_moves += self._brute_force_move(
                        int(self._position_of_qudit(ion_at_segment[0], pgs)),
                        int(available_spaces[0]),
                        pgs,
                        congestion_cache=congestion_cache,
                    )
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                    available_spaces = []
                    for trap in self.qccd_machine.physical_graph.trap_list:
                        _, available_space = self.qccd_machine.trap_is_fully_occupied_pgs(trap.id, pgs)
                        available_spaces += available_space
                    current_assignment = self._assignment_from_pgs(pgs)
                    ion_at_segment = []
                    for ion in current_assignment:
                        pos = self._position_of_qudit(ion, pgs)
                        if pos in self.qccd_machine.physical_to_position["segment_space"]:
                            ion_at_segment.append(ion)
                for ion_index in range(len(ion_order)):
                    gate_pos[ion_index] = self._position_of_qudit(ion_order[ion_index], pgs)
            """
                Clearing the end-point of the selected trap
            """
            if pos_idx == len(gate_pos) - 1:
                continue
            elif (self._is_occupied(int(selected_end_point[pos_idx + 1]), pgs) and
                  (pos_idx != len(gate_pos) - 1 and
                   self._logical_at_position(int(selected_end_point[pos_idx + 1]), pgs)
                   not in gate_pi)):
                self._status(f'Clearing endpoint {selected_end_point[pos_idx+1]}.........')
                end_point_neighbors = sorted(
                    self.qccd_machine.get_move_neighbors(selected_end_point[pos_idx + 1]),
                )
                # if any(position in end_point_neighbors for position in gate_pos):
                #     print(f"Not clearing endpoint as it affect the next gate position...")
                #     for ion_index in range(len(ion_order)):
                #         gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
                #     print(f"Gate position gate updated to {gate_pos}--------------c")
                #     continue
                occupied_neighbors = []
                for neighbor in end_point_neighbors:
                    if self._is_occupied(neighbor, pgs):
                        occupied_neighbors.append(neighbor)
                end_point_neighbors = [i for i in end_point_neighbors if i not in occupied_neighbors]
                if not end_point_neighbors:
                    potential_blockage = [i for i in occupied_neighbors if self.qccd_machine.get_trap_id(i) is None]
                    leading_moves += self._brute_force_move(
                        int(selected_end_point[pos_idx + 1]), int(potential_blockage[0]),
                        pgs, clearing_ep=True, congestion_cache=congestion_cache,
                    )
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                else:
                    self._apply_and_append_move(
                        leading_moves,
                        (selected_end_point[pos_idx + 1], end_point_neighbors[0]),
                        pgs,
                        context='clear endpoint neighbor',
                    )
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                    # print(f"Perform move {(selected_end_point[pos_idx + 1], end_point_neighbors[0])} to clear "
                    #       f"the endpoint")
            for ion_index in range(len(ion_order)):
                gate_pos[ion_index] = self._position_of_qudit(ion_order[ion_index], pgs)
            self._status(lambda: f'Gate position gate updated to {gate_pos}--------------c')
        self.last_bruteforce_trace['final_assignment'] = self._assignment_from_pgs(pgs)
        self.last_bruteforce_trace['congestion_cache_entries'] = len(congestion_cache)
        return leading_moves

    def _brute_force_move(
            self,
            position: int,
            trap_space: int,
            pgs: PositionGraphState,
            clearing_ep: bool = False,
            congestion_cache: dict[tuple[object, ...], tuple[float, float]] | None = None,
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        # print(
        #     f"Trying to move position {position} to trap space {trap_space} with ion assignment {ion_assignment}")
        leading_moves = []
        path = self.qccd_machine.get_move_path(position, trap_space)
        ion_status = self.qccd_machine.position_to_physical[position]
        for idx_point in range(len(path) - 1):
            possible_move = (path[idx_point], path[idx_point + 1])
            if not self._is_occupied(path[idx_point + 1], pgs):
                self._apply_and_append_move(
                    leading_moves,
                    possible_move,
                    pgs,
                    context='bruteforce empty neighbor',
                )
                # print(
                #     f"Perform move {(possible_move, ion_assignment)} as there is no ion in the neighbor, "
                #     f"ion status: {ion_status}")
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                self._apply_and_append_move(
                    leading_moves,
                    possible_move,
                    pgs,
                    context='bruteforce trap inner swap',
                )
                # print(f"Perform move {possible_move} with inner-swap, ion status: {ion_status}")
            else:
                ion_pos = path[idx_point]
                blockage = path[idx_point + 1]
                # print(f"There is blockage at {blockage}, try to resolve it...")
                if congestion_cache is None:
                    congestion_cache = {}
                leading_moves += self._resolve_congestion(
                    ion_pos,
                    path,
                    blockage,
                    pgs,
                    ion_pos,
                    blockage,
                    congestion_cache=congestion_cache,
                )
                self._apply_and_append_move(
                    leading_moves,
                    possible_move,
                    pgs,
                    context='bruteforce after congestion resolution',
                )
                # print(f"Perform move {possible_move} after resolving blockage")
                # print(f"Ion assignment after resolving blockage: {ion_assignment}")
            if ion_status == 'segment' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                ion_status = 'trap'
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'segment':
                ion_status = 'segment'
        return leading_moves

    def _resolve_congestion(
            self,
            target: int,
            path: list[int],
            blockage: int,
            pgs: PositionGraphState,
            original_target: int,
            original_blockage: int,
            num_call: int = 0,
            clearing_ep: bool = False,
            congestion_cache: dict[tuple[object, ...], tuple[float, float]] | None = None,
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        if num_call > 100:
            raise ValueError("Too many repetitive call...")
        if congestion_cache is None:
            congestion_cache = {}
        # print(
        #     f"Trying to resolve blockage {blockage} from the target {target} path with ion assignment {ion_assignment}")
        _logger.debug(
            f"Trying to resolve blockage {blockage} from the target {target} "
            f"path with {self._assignment_from_pgs(pgs)}"
        )
        leading_moves = []
        # print("Path: {}".format(path))
        # print("Original target: ", original_target)
        # print("Original blockage: ", original_blockage)
        target_ion_index = self._logical_at_position(target, pgs)
        blockage_neighbors = sorted(self.qccd_machine.get_move_neighbors(blockage))
        initial_blockage_neighbors = list(blockage_neighbors)
        # print("Initial blockage neighbors: ", blockage_neighbors)
        if original_target in blockage_neighbors:
            blockage_neighbors.remove(original_target)
        if original_blockage in blockage_neighbors:
            blockage_neighbors.remove(original_blockage)
        if target in blockage_neighbors:
            blockage_neighbors.remove(target)
        removed_blockage_neighbors = []
        #print("blockage neighbors: ", blockage_neighbors)
        for neighbor in blockage_neighbors:
            # print("Neighbor: {}".format(neighbor))
            if neighbor in path:
                removed_labeled = True
                if len(blockage_neighbors) == 1:
                    removed_labeled = False
                for neighbor_ext in self.qccd_machine.get_move_neighbors(neighbor):
                    if not self._is_occupied(neighbor_ext, pgs) and neighbor_ext not in path:
                        removed_labeled = False
                if removed_labeled:
                    removed_blockage_neighbors.append(neighbor)

        blockage_neighbors = [i for i in blockage_neighbors if i not in removed_blockage_neighbors]
        # print("Blockage_neighbors: ", blockage_neighbors)
        # _logger.debug(f"Blockage neighbors: {blockage_neighbors}")
        potential_blockage = []
        for neighbor in blockage_neighbors:
            if self._is_occupied(neighbor, pgs):
                potential_blockage.append(neighbor)
        for neighbor in potential_blockage:
            blockage_neighbors.remove(neighbor)
        resolve_entry: dict[str, object] = {
            'num_call': int(num_call),
            'target': int(target),
            'blockage': int(blockage),
            'path': [int(x) for x in path],
            'initial_blockage_neighbors': [int(x) for x in initial_blockage_neighbors],
            'filtered_blockage_neighbors': [int(x) for x in blockage_neighbors],
            'potential_blockage': [int(x) for x in potential_blockage],
            'clearing_ep': bool(clearing_ep),
        }
        # _logger.debug(f"Potential blockage neighbors: {potential_blockage}")
        # print(f"Updated Blockage neighbors: {blockage_neighbors}")
        # print(f"Potential blockage neighbors: {potential_blockage}")
        # Todo: Instead of simply use the first element, can we do sth better? (DONE)
        if blockage_neighbors:
            congestion = np.array([
                self._cached_congestion_rate(
                    blockage_neighbor,
                    target,
                    blockage,
                    pgs,
                    depth=self.qccd_machine.max_ion_capacity - 1 + num_call,
                    cache=congestion_cache,
                )
                for blockage_neighbor in blockage_neighbors
            ])
            _logger.debug(f"Congestion: {congestion}")
            congestion_rates = congestion[:, 0]
            congestion_scores = congestion[:, 1]
            # print("Cogestion rates: ", congestion_rates)
            choosen_indices = list(np.where(congestion_rates == np.min(congestion_rates))) #int(np.argmin(cogestion_rates))
            # print("Choosen indices:", choosen_indices[0])
            # print("Len choosen indices: ", len(choosen_indices[0]))
            # print("Choosen indices types: ", len(choosen_indices[0]))
            if len(choosen_indices[0]) > 1:
                # print("Congestion scores: ", congestion_scores)
                choosen_idx = int(np.argmin(congestion_scores[choosen_indices[0]]))
            else:
                choosen_idx = int(choosen_indices[0][0])
            resolve_entry['branch'] = 'free_neighbor'
            resolve_entry['congestion_rates'] = [float(x) for x in congestion_rates]
            resolve_entry['congestion_scores'] = [float(x) for x in congestion_scores]
            resolve_entry['chosen_neighbor'] = int(blockage_neighbors[choosen_idx])
            self._append_resolve_trace(resolve_entry)
            # print(f"Choose to resolve {blockage_neighbors[choosen_idx]}")
            self._apply_and_append_move(
                leading_moves,
                (blockage, blockage_neighbors[choosen_idx]),
                pgs,
                context=f'resolve free neighbor num_call={num_call}',
            )
            # print(f"Blockage: {blockage}, blockage neighbors: {blockage_neighbors[choosen_idx]}")
            # print(
            #     f"Perform move (1) {(blockage, blockage_neighbors[choosen_idx])} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            return leading_moves
        elif potential_blockage:
            congestion = np.array([
                self._cached_congestion_rate(
                    blockage_neighbor,
                    target,
                    blockage,
                    pgs,
                    depth=self.qccd_machine.max_ion_capacity - 1 + num_call,
                    cache=congestion_cache,
                )
                for blockage_neighbor in potential_blockage
            ])
            congestion_rates = congestion[:, 0]
            congestion_scores = congestion[:, 1]
            # print(f"Congestion rates: {congestion_rates}")
            if len(congestion_rates) == 0:
                congestion_rates = [1.0]
            choosen_indices = list(np.where(congestion_rates == np.min(congestion_rates))) #int(np.argmin(cogestion_rates))
            # print("Choosen indices:", choosen_indices[0])
            # print("Len choosen indices: ", len(choosen_indices[0]))
            if len(choosen_indices[0]) > 1:
                # print("Congestion score: ", congestion_scores)
                choosen_idx = int(np.argmin(congestion_scores[choosen_indices[0]]))
            else:
                choosen_idx = int(choosen_indices[0][0])
            resolve_entry['branch'] = 'potential_blockage'
            resolve_entry['congestion_rates'] = [float(x) for x in congestion_rates]
            resolve_entry['congestion_scores'] = [float(x) for x in congestion_scores]
            resolve_entry['chosen_neighbor'] = int(potential_blockage[choosen_idx])

            if congestion_rates[choosen_idx] == 1.0 and clearing_ep is False:
                resolve_entry['fallback'] = 'reverse_target'
                self._append_resolve_trace(resolve_entry)
                # print(f"As the best path leads to deadend, we choose to re-add the target to potential neighbor")
                if self._cached_congestion_rate(
                    target,
                    target,
                    blockage,
                    pgs,
                    depth=self.qccd_machine.max_ion_capacity - 1 + num_call,
                    cache=congestion_cache,
                )[0] <= congestion_rates[choosen_idx]:
                    # Reverse move (treat target as blockage and vice versa)
                    # print(f"Blockage: {blockage}, target: {target}")
                    leading_moves += self._resolve_congestion(blockage, [], target, pgs,
                                                                    original_target, original_blockage, num_call + 1,
                                                                    congestion_cache=congestion_cache)
                    self._apply_and_append_move(
                        leading_moves,
                        (blockage, target),
                        pgs,
                        context=f'resolve reverse target step 1 num_call={num_call}',
                    )
                    # print(
                    #     f"Perform move (2) {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                    blockage = target
                    target = self._position_of_qudit(target_ion_index, pgs)
                    # print(f"Blockage: {blockage}, target: {target}")
                    if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                            and self.qccd_machine.get_trap_id(blockage) is None):
                        leading_moves += self._resolve_congestion(target, [], blockage, pgs,
                                                                  original_target, original_blockage, num_call+1,
                                                                  congestion_cache=congestion_cache)
                    self._apply_and_append_move(
                        leading_moves,
                        (blockage, target),
                        pgs,
                        context=f'resolve reverse target step 2 num_call={num_call}',
                    )
                    # print(
                    #     f"Perform move (2') {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                else:
                    raise ValueError("This method does not resolve this case !!!")
            else:
                self._append_resolve_trace(resolve_entry)
                #print(f"Choose to resolve {potential_blockage[choosen_idx]}")
                leading_moves += self._resolve_congestion(blockage, path, potential_blockage[choosen_idx],
                                                          pgs, original_target, original_blockage,
                                                          num_call + 1,
                                                          congestion_cache=congestion_cache)
                self._apply_and_append_move(
                    leading_moves,
                    (blockage, potential_blockage[choosen_idx]),
                    pgs,
                    context=f'resolve potential blockage num_call={num_call}',
                )
                # print(f"Blockage: {blockage}, potential blockage: {potential_blockage[choosen_idx]}")
                # print(
                #     f"Perform move (3) {(blockage, potential_blockage[choosen_idx])} to try resolving the blockage "
                #     f"at {blockage} as we have moved the target ions.")
                # print("Current ion assignment: ", ion_assignment)
        else:
            resolve_entry['branch'] = 'deadend'
            self._append_resolve_trace(resolve_entry)
            # print("No blockage neighbors...")
            # print(f"As the best path leads to deadend, we choose to re-add the target to potential neighbor")
            # print(f"Blockage: {blockage}, target: {target}")
            leading_moves += self._resolve_congestion(blockage, [], target, pgs,
                                                      original_target, original_blockage, num_call+1,
                                                      congestion_cache=congestion_cache)
            self._apply_and_append_move(
                leading_moves,
                (blockage, target),
                pgs,
                context=f'resolve deadend step 1 num_call={num_call}',
            )
            # print(
            #     f"Perform move (4) {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            blockage = target
            target = self._position_of_qudit(target_ion_index, pgs)
            # print(f"Blockage: {blockage}, target: {target}")
            if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                    and self.qccd_machine.get_trap_id(blockage) is None):
                leading_moves += self._resolve_congestion(target, [], blockage, pgs,
                                                          original_target, original_blockage, num_call+1,
                                                          congestion_cache=congestion_cache)
            self._apply_and_append_move(
                leading_moves,
                (blockage, target),
                pgs,
                context=f'resolve deadend step 2 num_call={num_call}',
            )
            # print(
            #     f"Perform move (4') {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
        return leading_moves

    def congestion_rate(self,
                        position: int,
                        target: int,
                        blockage: int,
                        pgs: PositionGraphState,
                        depth: int = 2):
        layers = self._congestion_layers(position, target, blockage, depth)
        return self._congestion_rate_from_layers(position, layers, pgs)

    ######################################################################################
    def backward_pass(
            self,
            circuit: Circuit,
            pgs: PositionGraphState,
    ) -> None:
        """
        Apply a backward pass of the QCCD mapper using PositionGraphState.

        Args:
            circuit (Circuit): The circuit to pass over.

            pgs (PositionGraphState): The live logical-to-position state.
        """
        # Preprocessing
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.rear
        decay = [1.0 for _ in range(self.qccd_machine.num_positions)]
        repeated_path = False
        executed_flag = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        leading_moves: list[tuple[int, int]] = []
        heuristic_move = True
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                '%s Starting backward sabre QCCD pass with ion assignment: %s.',
                self._log_prefix,
                self._assignment_from_pgs(pgs),
            )
        longest_path = self.qccd_machine.get_longest_move_path_length()
        # Main Loop
        while len(F) > 0:
            # print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                # print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path/4):
                # print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                leading_moves += self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pgs,
                )
            execute_list = self._sorted_points(
                [n for n in F if self.qccd_machine.gate_is_executable_pgs(circuit[n], pgs)],
            )
            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                # Rest penalty from reptition
                self.extended_set_size = initial_extended_set_size
                # Add the temporary F to current F
                if tmp_F:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    next_executed_counts.pop(n)
                    _logger.debug('%s Executing gate at point %s.', self._log_prefix, n)
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

            executed_flag = False
            # Pick and apply a swap
            if repeated_path:
                # If there is repetition, first take into account only one gate in F
                # Then change the extended set size to 0
                repeated_path = False
                if len(F) == 1 and self.extended_set_size != 0:
                    self.extended_set_size = 0
                elif len(F) == 1 and self.extended_set_size == 0:
                    # Retrieve executable gates giving the current mapping `pi`
                    if self.iter_count > 2:
                        # print("Try bruteforce due to repeated pattern...")
                        leading_moves += self._brute_force_congestion(
                            circuit[self._sorted_points(F)[0]], D, pgs,
                        )
                        continue
                else:
                    ordered_F = self._sorted_points(F)
                    tmp_F = ordered_F[1:]
                    F = [ordered_F[0]]
                    #print(f"Front is modified to {F}.")

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            #print(f"Extended set: {[circuit[n] for n in E]}")
            if self.force_bruteforce:
                best_move = None
            else:
                best_move = self._get_best_move(
                    circuit, F, E, D, decay, heuristic_move, pgs,
                )
            if best_move is None:
                leading_moves += self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pgs,
                )
                continue
            # print(f"Best move: {best_move}")
            self._apply_move(best_move, pgs=pgs)
            leading_moves.append(best_move)
            self.iter_count += 1

    def _calc_extended_set(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
    ) -> set[CircuitPoint]:
        """Calculate the Extended Set for look-ahead capabilities."""
        extended_set: set[CircuitPoint] = set()
        frontier = self._sorted_points(copy.copy(F))
        while len(frontier) > 0 and len(extended_set) < self.extended_set_size:
            n = frontier.pop(0)
            next_points = self._sorted_points(circuit.next(n))
            extended_set.update(next_points)
            frontier.extend(next_points)
        return extended_set

    def _get_best_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            E: set[CircuitPoint],
            D: list[list[float]],
            decay: list[float],
            heuristic_move: bool,
            pgs: PositionGraphState,
    ) -> tuple[int, int]:
        """Return the best move given the current algorithm state and ion assignment. (Logical function)"""
        # Track best one
        best_score = np.inf
        best_move = None

        # Gather all considerable moves
        if heuristic_move:
            move_candidate_list = self._obtain_heuristic_moves(
                circuit,
                F,
                pgs,
            )
        else:
            move_candidate_list = self._obtain_moves(circuit, pgs)
        frontier_points = self._sorted_points(F)
        frontier_locations = [tuple(int(q) for q in circuit[n].location) for n in frontier_points]
        if self._debug_compare_enabled():
            self._debug_compare(
                lambda: (
                    f'frontier points={frontier_points} '
                    f'logical_locations={frontier_locations}'
                ),
            )
            self._debug_compare(
                lambda: (
                    f'current assignment='
                    f'{dict(sorted(self._assignment_from_pgs(pgs).items()))}'
                ),
            )
            self._debug_compare(
                lambda: f'candidate moves={self._sorted_moves(move_candidate_list)}',
            )
        list_of_best_move = []
        move_scores = []
        # Score them, tracking the best one
        # scores = Parallel(n_jobs=5)(delayed(self._score_move)(circuit, F, D, pi, ion_assignment, move, decay, E)
        #                             for move in move_candidate_list)
        # list_of_best_score = np.argwhere(scores == np.max(scores)).flatten().tolist()
        # list_of_best_moves = list(move_candidate_list)[list_of_best_score]
        for move in self._sorted_moves(move_candidate_list):
            score = self._score_move(
                circuit,
                F,
                D,
                move,
                decay,
                E,
                pgs,
            )
            move_scores.append((move, float(score)))
            if score < best_score:
                best_score = score
                best_move = move
                list_of_best_move = [move]
            elif score == best_score:
                list_of_best_move.append(move)
        self._debug_compare(lambda: f'move scores={move_scores}')
        if best_move is None:
            # print("*** Unable to find best move. ***")
            return None
            # raise RuntimeError('Unable to find best move.')
        # print(f"List of best move: {list_of_best_move}")
        if len(list_of_best_move) == 1:
            self._debug_compare(
                lambda: f'chosen move={best_move} best_score={float(best_score)}',
            )
            return best_move
        else:
            # ToDo: There is some case where we have to decide between moves with
            #  same score, we choose move with the most potential influence... (Done)
            move_relative_scores = []
            for move in list_of_best_move:
                move_relative_score = 0.0
                if D[move[0]][move[1]] == self.qccd_machine.timing_data['merge']:
                    move_relative_score = -self.qccd_machine.timing_data['merge']
                for n in F:
                    location = circuit[n].location
                    p = [self._position_of_qudit(loc, pgs) for loc in location]
                    if any([pos in move for pos in p]):
                        move_relative_score = 0.0
                    else:
                        move_relative_score += np.sum([np.min([D[pos][move[0]], D[pos][move[1]]]) for pos in p])
                move_relative_scores.append(move_relative_score)
            tied_moves = [
                list_of_best_move[i]
                for i in np.where(move_relative_scores == np.min(move_relative_scores))[0]
            ]
            chosen = self._sorted_moves(tied_moves)[0]
            self._debug_compare(
                'tie-break among best moves='
                + str(list_of_best_move)
                + ' relative_scores='
                + str([float(score) for score in move_relative_scores])
                + f' chosen={chosen}',
            )
            return chosen

    def _obtain_heuristic_moves(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            pgs: PositionGraphState,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves w.r.t frontier given the current QCCD hardware."""
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = []
        for location in list(F)[:1]:
            block = circuit[location]
            for qudit in block.location:
                physical_qudit_positions.append(self._position_of_qudit(qudit, pgs))
        physical_qudit_positions = self._sorted_unique_ints(physical_qudit_positions)
        moves = set()
        for physical_qudit_position in physical_qudit_positions:
            neighbors = sorted(self.qccd_machine.get_move_neighbors(physical_qudit_position))
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if self._is_occupied(a, pgs) and self._is_occupied(b, pgs):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _obtain_moves(
            self,
            circuit: Circuit,
            pgs: PositionGraphState,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves given the current QCCD hardware."""
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = [
            self._position_of_qudit(i, pgs) for i in circuit.active_qudits
        ]
        physical_qudit_positions = self._sorted_unique_ints(physical_qudit_positions)
        moves = set()
        for physical_qudit_position in physical_qudit_positions:
            neighbors = sorted(self.qccd_machine.get_move_neighbors(physical_qudit_position))
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if self._is_occupied(a, pgs) and self._is_occupied(b, pgs):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _score_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            D: list[list[float]],
            move: tuple[int, int],
            decay: list[float],
            E: set[CircuitPoint],
            pgs: PositionGraphState,
    ) -> float:
        """Score the candidate realizable physical moves given the current algorithm state and ion assignment."""
        l1 = self._logical_at_position(move[0], pgs)
        l2 = self._logical_at_position(move[1], pgs)
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
        affected = [logical for logical in (l1, l2) if logical is not None]
        snapshot = self._snapshot_logical_positions(pgs, affected)
        try:
            self._apply_move(move, pgs=pgs)

            # Calculate front set term
            front = 0.0
            for n in F:
                logical_qudits = circuit[n].location
                front += self._get_distance(logical_qudits, D, pgs)
            front /= len(F)

            # Calculate extended set term
            extend = 0.0
            # Match the legacy CG implementation for apples-to-apples comparison.
            # if len(E) > 0:
            #     for n in E:
            #         extend += self._get_distance(circuit[n].location, D, pgs)
            #     extend /= len(E)
            #     extend *= self.extended_set_weight

            return front + extend
        finally:
            self._restore_logical_positions(pgs, snapshot)

    # def _get_distance_from_position_to_trap(self,
    #                                         position: int,
    #                                         available_space: list[int],
    #                                         D: list[list[float]],
    #                                         pi: dict) -> float:
    #     distance = np.inf
    #     for space in available_space:
    #         # ToDo: Find a way to cooperate the penalty
    #         #  (The penalty is the cost to resolve not able to get to the trap)
    #         #  of not able to move to a trap to the  distance wise (when calculating the min) (DONE)
    #
    #         _, block_w = self.qccd_machine.path_is_blocked(position, space, pi)
    #         # print(f"Number of block in path {position, space}: {block_w}")
    #         space_distance = D[position][space]
    #         for block_position in block_w:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 # print("Minimum distance to endpoints: ", min_to_endpoints)
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         # print(f"Distance when considering space {space}: ", space_distance)
    #         distance = np.min([space_distance,
    #                            distance])
    #     return distance

    # def _get_distance_from_two_position_to_trap(self,
    #                                             positions: list[int],
    #                                             available_space: list[int],
    #                                             D: list[list[float]],
    #                                             pi: dict) -> (float, float):
    #     # ToDo: If two point refer to the same point on the same trap, this create
    #     # local min situation and we need to modify this (2 point to 2 point on the same trap)
    #     distance = np.inf
    #     #print("Available space: ", available_space)
    #     for space in permutations(available_space, 2):
    #         #print("Considering space combination: {}".format(space))
    #         _, block_w_0 = self.qccd_machine.path_is_blocked(positions[0], space[0], pi)
    #         _, block_w_1 = self.qccd_machine.path_is_blocked(positions[1], space[1], pi)
    #         #print(f"Number of block in path {positions[0], space[0]}: {block_w_0}")
    #         #print(f"Number of block in path {positions[1], space[1]}: {block_w_1}")
    #         space_distance = D[positions[0]][space[0]] + D[positions[1]][space[1]]
    #         #print("Space distance: ", space_distance)
    #         for block_position in block_w_0:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         for block_position in block_w_1:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         distance = np.min([space_distance,
    #                            distance])
    #         # print(f"Distance when considering space {space}: ", space_distance)
    #         # print(f"Current minimum distance: ", distance)
    #     return distance / 2, distance / 2
    def _get_distance_from_position_to_trap(
            self,
            positions: list[int],
            available_space: list[int],
            D: list[list[float]],
            pgs: PositionState | None = None,
    ) -> float:
        """
            Get minimum distance from all the position of the gate to the trap...
        """
        distance = np.inf
        """
            All position of trap are considered...
        """
        #print("Available space: ", available_space)
        if pgs is None:
            raise ValueError('PositionGraphState is required for native distance scoring.')
        for space in permutations(available_space, len(positions)):
            space_distance = float(
                sum(D[positions[i]][space[i]] for i in range(len(positions)))
            )
            for i in range(len(positions)):
                blockage_profile = self.qccd_machine.get_move_blockage_profile(
                    positions[i],
                    space[i],
                )
                for block_position, resolve_cost in blockage_profile:
                    if pgs.get_logical_qudit_at_position(block_position) != -1:
                        space_distance += resolve_cost
            # print(f"Distance when considering space {space}: ", space_distance)
            # print(f"Current minimum distance: ", distance)
            # print("........")
            distance = min(space_distance, distance)
        """
            Only consider the endpoint
        """
        # for space in available_space:
        #     blockage = [self.qccd_machine.path_is_blocked(positions[i], space, ion_assignment)[1]
        #                 for i in range(len(positions))]
        #     space_distance = np.sum([D[positions[i]][space] for i in range(len(positions))])
        #     for block_w in blockage:
        #         for block_position in block_w:
        #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
        #                 resolve_cost = self.qccd_machine.timing_data['junction_Y']
        #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
        #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
        #                                            self.qccd_machine.trap_end_points[
        #                                                self.qccd_machine.get_trap_id(block_position)]])
        #                 resolve_cost = (self.qccd_machine.timing_data['split'] + min_to_endpoints)
        #             else:
        #                 raise ValueError("The block position is undefined as it sit on ",
        #                                  self.qccd_machine.position_to_physical[block_position])
        #             space_distance += resolve_cost
        #     distance = np.min([space_distance, distance])
        return distance

    def _get_distance(
            self,
            logical_qudits: Sequence[int],
            D: list[list[float]],
            pgs: PositionGraphState,
    ) -> float:
        """
            Calculate the expected cost w.r.t distance to connect logical qudits.
        """
        # Single qudit case
        if len(logical_qudits) == 1:
            p = [self._position_of_qudit(logical_qudits[0], pgs)]
            trap_p = self.qccd_machine.get_trap_id(p[0])
            if trap_p is not None:
                return 0.0
            else:
                distance_to_trap = np.inf
                for trap in self.qccd_machine.physical_graph.executable_trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied_pgs(trap.id, pgs)
                    #endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
                    # Change to endpoints of trap space ... TODO
                    distance_to_trap = np.min([self._get_distance_from_position_to_trap(p, available_space, D, pgs),
                                               distance_to_trap])
                return distance_to_trap
        # Multi-qudit case
        p_list = [self._position_of_qudit(logical_qudit, pgs) for logical_qudit in logical_qudits]
        pairwise_distance = [D[p1][p2] for p1, p2 in combinations(p_list, 2)]
        distance = np.max(pairwise_distance)
        # Distance to nearest trap from pg
        trap_p = [self.qccd_machine.get_trap_id(p) for p in p_list]
        if trap_p.count(None) == 0 and trap_p.count(trap_p[0]) == len(trap_p):
            total_F = 0.0
        else:
            total_F = np.inf
            for trap in self.qccd_machine.physical_graph.executable_trap_list:
                #endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
                _, available_space = self.qccd_machine.trap_is_fully_occupied_pgs(trap.id, pgs)
                considering_p = []
                for trap_p_index, p in zip(trap_p, p_list):
                    if trap_p_index == trap.id:
                        continue
                    else:
                        considering_p.append(p)
                if not considering_p:
                    considering_dist_to_F = 0.0
                else:
                    # Change to endpoints of trap space ... TODO
                    considering_dist_to_F = self._get_distance_from_position_to_trap(
                        considering_p,
                        available_space,
                        D,
                        pgs,
                    )
                total_F = np.min([total_F, considering_dist_to_F])
        # print(
        #     f"Physical distance w.r.t gate {logical_qudits} is {distance} "
        #     f"Total distance to nearest similar trap: {total_F}"
        # )
        return distance + total_F

    def _apply_move(
            self,
            move: tuple[int, int],
            pgs: PositionState,
    ) -> bool:
        """Apply the move to the live PositionGraphState."""
        move = (int(move[0]), int(move[1]))
        if move[0] == move[1]:
            _logger.debug('skipping self move %s', move)
            return False
        _logger.debug('applying move %s' % str(move))
        l1 = self._logical_at_position(move[0], pgs)
        l2 = self._logical_at_position(move[1], pgs)
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
        if l1 is None:
            pgs.set_qudit_position(l2, move[0])
        elif l2 is None:
            pgs.set_qudit_position(l1, move[1])
        else:
            pgs.swap_logical_qudits(l1, l2)
        _logger.debug('ion assignment after move %s', self._assignment_from_pgs(pgs))
        return True
        # decay[move[0]] += self.decay_delta
        # decay[move[1]] += self.decay_delta

    # def _uphill_swaps(
    #     self,
    #     logical_qudits: Sequence[int],
    #     cg: CouplingGraph,
    #     pi: list[int],
    #     D: list[list[int]],
    # ) -> Iterator[tuple[int, int]]:
    #     """Yield the swaps necessary to bring some of the qudits together."""
    #     center_qudit = min(
    #         logical_qudits,
    #         key=lambda q: sum(
    #             D[pi[q]][pi[p]]
    #             for p in logical_qudits
    #             if p != q
    #         ),
    #     )
    #
    #     for q in logical_qudits:
    #         if q == center_qudit:
    #             continue
    #
    #         # TODO: Do not need to calculate entire tree
    #         spt = cg.get_shortest_path_tree(pi[center_qudit])
    #         path = list(reversed(spt[pi[q]]))
    #
    #         _logger.debug(f'Moving {q} to {center_qudit} via {path}.')
    #
    #         for p1, p2 in zip(path, path[1:]):
    #             if pi[center_qudit] == p1 or pi[center_qudit] == p2:
    #                 continue
    #             yield (p1, p2)

    def _apply_perm(self, perm: Sequence[int], pi: list[int], ion_assignment: dict) -> None:
        """Apply the `perm` permutation to the current mapping `pi`."""
        _logger.debug('initial pi %s' % str(pi))
        _logger.debug('initial ion_assignment %s' % str(ion_assignment))
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]
        ion_assignment_tmp = ion_assignment.copy()
        for p in ion_assignment.keys():
            ion_assignment[p] = ion_assignment_tmp[pi.index(p)]
        _logger.debug('ion assignment after permutation %s ' % str(ion_assignment))


if __name__ == '__main__':
    from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
    from bqskit import Circuit

    physical_model = create_testing_physical_machine()
    timing_data = {'sq_timings': 30e-6,
                   'tq_timings': 40e-6,
                   'segment': 5e-6,
                   'inner_swap': 42e-6,
                   'split': 80e-6,
                   'merge': 80e-6,
                   'junction_Y': 100e-6,
                   'junction_X': 120e-6}
    machine_model = QCCDMachineModel(physical_graph=physical_model,
                                     multi_qudit_gate_type='FM',
                                     timing_data=timing_data)
    ion_assignment = {0: 0, 1: 1, 2: 2,
                      3: 6, 4: 10}
    circuit = Circuit.from_file("data/input_qasms/Grover_5.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_5.qasm")
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 6, 4: 10, 5: 11,
    #                   6: 7, 7: 9}
    # circuit = Circuit.from_file("data/input_qasms/Grover_8.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_8.qasm")
    pi = [i for i in range(5)]
    # ion_assignment = {0: 6, 1: 2, 2: 0,
    #               3: 1, 4: 5, 5: 3,
    #               6: 7, 7: 8, 8: 4}
    # circuit = Circuit.from_file("data/input_qasms/adder9_trapsize3.qasm")
    # circuit = Circuit(5)
    # circuit.append_gate(CNOTGate(), (0, 1))
    # circuit.append_gate(CNOTGate(), (1, 2))

    mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                        decay_delta=0.0,
                                        extended_set_size=5,
                                        extended_set_weight=0.5)
    pgs = machine_model.build_pgs_from_assignment(ion_assignment)
    mapping_algo.forward_pass(circuit, pgs=pgs, modify_circuit=True)
