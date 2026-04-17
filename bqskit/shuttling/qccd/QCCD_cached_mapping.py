"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations
import math
import copy
import logging
import itertools
import os
from typing import Iterator
from typing import Sequence
from itertools import permutations, combinations
import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.shuttling.qccd.QCCD_cached_machine import QCCDMachineModel
from bqskit.ir.gates.barrier import BarrierPlaceholder

_logger = logging.getLogger(__name__)


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
            cogestion_rate: float = 0.85,
            force_bruteforce: bool = True,
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
        self.distance_stats: dict[str, int] = {
            'calls': 0,
            'single_qudit_calls': 0,
            'multi_qudit_calls': 0,
        }

    def _sorted_points(self, points) -> list[CircuitPoint]:
        return sorted(points, key=lambda p: (p.cycle, p.qudit))

    def _sorted_unique_ints(self, values) -> list[int]:
        return sorted(set(int(v) for v in values))

    def _sorted_moves(self, moves) -> list[tuple[int, int]]:
        return sorted((int(a), int(b)) for a, b in moves)

    def _logical_at_position_from_assignment(
        self,
        pos: int,
        ion_assignment: dict[int, int],
    ) -> int | None:
        for logical, position in ion_assignment.items():
            if int(position) == pos:
                return int(logical)
        return None

    def _canonicalize_move_for_assignment(
        self,
        move: tuple[int, int],
        ion_assignment: dict[int, int],
    ) -> tuple[int, int] | None:
        u, v = (int(move[0]), int(move[1]))
        logical_u = self._logical_at_position_from_assignment(u, ion_assignment)
        logical_v = self._logical_at_position_from_assignment(v, ion_assignment)
        if logical_u is None and logical_v is None:
            return None
        if logical_u is None:
            return (v, u)
        return (u, v)

    def _append_logged_move_instruction(
        self,
        instructions_list: list[list[str]],
        move: tuple[int, int],
        ion_assignment: dict[int, int],
        D: list[list[float]],
    ) -> bool:
        canonical_move = self._canonicalize_move_for_assignment(move, ion_assignment)
        if canonical_move is None:
            return False
        self._apply_move(canonical_move, ion_assignment)
        instructions_list.append([
            f"Move {canonical_move}",
            f"{dict(ion_assignment)}",
            f"cost: {D[canonical_move[0]][canonical_move[1]]} seconds",
        ])
        return True

    @property
    def _log_prefix(self) -> str:
        return '[CG]'

    def _status(self, message: str) -> None:
        if os.getenv('BQSKIT_QCCD_VERBOSE', '1').lower() not in (
            '0', 'false', 'no', 'off',
        ):
            print(f'{self._log_prefix} {message}')

    def _debug_compare_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_COMPARE_DEBUG', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _debug_compare(self, message: str) -> None:
        if self._debug_compare_enabled():
            print(f'{self._log_prefix}[compare] {message}')

    def _capture_forward_trace_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_CAPTURE_TRACE', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _forward_probe_target(self) -> tuple[int, int] | None:
        forward_pass = os.getenv('BQSKIT_QCCD_PROBE_FORWARD_PASS')
        forward_step = os.getenv('BQSKIT_QCCD_PROBE_FORWARD_STEP')
        if forward_pass is None or forward_step is None:
            return None
        try:
            return int(forward_pass), int(forward_step)
        except ValueError:
            return None

    def _forward_probe_enabled(self) -> bool:
        return self._forward_probe_target() is not None

    def _forward_probe_verbose_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_PROBE_VERBOSE', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _capture_backward_trace_enabled(self) -> bool:
        return (
            os.getenv('BQSKIT_QCCD_CAPTURE_BACKWARD_TRACE', '').lower() in (
                '1', 'true', 'yes', 'on',
            )
            or os.getenv('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER', '').lower() in (
                '1', 'true', 'yes', 'on',
            )
        )

    def _backward_probe_target(self) -> int | None:
        backward_pass = os.getenv('BQSKIT_QCCD_PROBE_BACKWARD_PASS')
        if backward_pass is None:
            return None
        try:
            return int(backward_pass)
        except ValueError:
            return None

    def _backward_execute_probe_target(self) -> tuple[int, ...] | None:
        raw_value = os.getenv('BQSKIT_QCCD_PROBE_BACKWARD_EXECUTE_GATE')
        if not raw_value:
            return None
        try:
            return tuple(
                int(part.strip())
                for part in raw_value.split(',')
                if part.strip() != ''
            )
        except ValueError:
            return None

    def _backward_targeted_probe_enabled(self) -> bool:
        return (
            self._backward_probe_matches()
            and self._backward_execute_probe_target() is not None
        )

    def _backward_target_gate_matches(
        self,
        gate_location: Sequence[int],
    ) -> bool:
        target_gate = self._backward_execute_probe_target()
        if target_gate is None or not self._backward_probe_matches():
            return False
        return tuple(int(q) for q in gate_location) == target_gate

    def _backward_probe_matches(self) -> bool:
        target = self._backward_probe_target()
        if target is None:
            return False
        active_pass = getattr(self, '_active_backward_pass_index', None)
        return active_pass == target

    def _backward_execute_probe_matches(
        self,
        circuit: Circuit,
        execute_points: Sequence[CircuitPoint],
    ) -> bool:
        target_gate = self._backward_execute_probe_target()
        if target_gate is None or not self._backward_probe_matches():
            return False
        execute_locations = self._format_locations(circuit, execute_points)
        return target_gate in execute_locations

    def _backward_gate_probe_matches_front(
        self,
        circuit: Circuit,
        front_points: Sequence[CircuitPoint],
    ) -> bool:
        target_gate = self._backward_execute_probe_target()
        if target_gate is None or not self._backward_probe_matches():
            return False
        front_locations = self._format_locations(circuit, front_points)
        return target_gate in front_locations

    def _emit_backward_probe(
        self,
        label: str,
        payload: object,
    ) -> None:
        if not self._backward_probe_matches():
            return
        if self._backward_targeted_probe_enabled():
            return
        if label == 'loop' and isinstance(payload, dict):
            if not payload.get('execute_locations'):
                return
        if label == 'best-move' and isinstance(payload, dict):
            if payload.get('result') is None:
                return
        print(f'{self._log_prefix}[backward-probe] {label}: {payload}')

    def _record_backward_trace(self, entry: dict[str, object]) -> None:
        if not self._capture_backward_trace_enabled():
            return
        trace = getattr(self, 'last_backward_trace', None)
        if not isinstance(trace, list):
            self.last_backward_trace = []
            trace = self.last_backward_trace
        trace.append(self._snapshot_trace_value(entry))

    def _assignment_delta(
        self,
        before: dict[int, int],
        after: dict[int, int],
    ) -> dict[int, tuple[int | None, int | None]]:
        delta: dict[int, tuple[int | None, int | None]] = {}
        for logical in sorted(set(before) | set(after)):
            before_value = before.get(logical)
            after_value = after.get(logical)
            if before_value != after_value:
                delta[int(logical)] = (
                    None if before_value is None else int(before_value),
                    None if after_value is None else int(after_value),
                )
        return delta

    def _emit_backward_execute_probe(
        self,
        payload: dict[str, object],
    ) -> None:
        if not self._backward_probe_matches():
            return
        if self._backward_execute_probe_target() is None:
            return
        print(f'{self._log_prefix}[backward-exec-probe] {payload}')

    def _emit_backward_bruteforce_probe(
        self,
        payload: dict[str, object],
    ) -> None:
        if not self._backward_probe_matches():
            return
        if self._backward_execute_probe_target() is None:
            return
        print(f'{self._log_prefix}[backward-bruteforce-probe] {payload}')

    def _emit_backward_bruteforce_trap_probe(
        self,
        payload: dict[str, object],
    ) -> None:
        if not self._backward_probe_matches():
            return
        if self._backward_execute_probe_target() is None:
            return
        print(f'{self._log_prefix}[backward-bruteforce-trap-probe] {payload}')

    def _snapshot_trace_value(self, value: object) -> object:
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

    def _forward_probe_matches(self, step_index: int) -> bool:
        target = self._forward_probe_target()
        if target is None:
            return False
        active_pass = getattr(self, '_active_forward_pass_index', None)
        return active_pass == target[0] and int(step_index) == target[1]

    def _active_forward_probe_step_matches(self) -> bool:
        return self._forward_probe_matches(
            int(getattr(self, '_active_forward_trace_step_index', 0)),
        )

    def _build_forward_probe_summary(
        self,
        *,
        circuit: Circuit,
        front_points: Sequence[CircuitPoint],
        execute_list: Sequence[CircuitPoint],
        action: str,
        step_index: int,
        best_move: tuple[int, int] | None,
        brute_force_gate: tuple[int, ...] | None,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            'forward_pass': int(getattr(self, '_active_forward_pass_index', -1)),
            'step_index': int(step_index),
            'action': action,
            'front_locations': self._format_locations(circuit, front_points),
            'execute_locations': self._format_locations(circuit, execute_list),
        }
        if best_move is not None:
            summary['best_move'] = tuple(int(x) for x in best_move)
        if brute_force_gate is not None:
            summary['brute_force_gate'] = tuple(int(x) for x in brute_force_gate)

        brute_force_trace = getattr(self, 'last_bruteforce_trace', None)
        if action != 'bruteforce' or not isinstance(brute_force_trace, dict):
            summary['brute_force_trace_present'] = False
            return summary

        summary['brute_force_trace_present'] = True
        summary['selected_trap_id'] = brute_force_trace.get('selected_trap_id')
        summary['selected_end_point'] = brute_force_trace.get('selected_end_point')
        summary['ion_order'] = brute_force_trace.get('ion_order')
        leading_moves = brute_force_trace.get('leading_moves', [])
        summary['leading_moves_count'] = len(leading_moves)

        segment_entries = brute_force_trace.get('segment_drain_trace', [])
        summary['segment_drain_triggered'] = bool(segment_entries)
        if segment_entries:
            first_segment = segment_entries[0]
            summary['segment_drain_first'] = {
                'phase': first_segment.get('phase'),
                'pos_idx': first_segment.get('pos_idx'),
                'ion_at_segment': first_segment.get('ion_at_segment'),
                'available_spaces': first_segment.get('available_spaces'),
            }

        resolve_entries = brute_force_trace.get('resolve_trace', [])
        summary['resolve_trace_count'] = len(resolve_entries)
        if resolve_entries:
            first_resolve = resolve_entries[0]
            summary['resolve_first'] = {
                'branch': first_resolve.get('branch'),
                'chosen_neighbor': first_resolve.get('chosen_neighbor'),
                'fallback': first_resolve.get('fallback'),
                'num_call': first_resolve.get('num_call'),
                'target': first_resolve.get('target'),
                'blockage': first_resolve.get('blockage'),
                'filtered_blockage_neighbors': first_resolve.get('filtered_blockage_neighbors'),
                'potential_blockage': first_resolve.get('potential_blockage'),
            }

        return summary

    def _emit_forward_probe(
        self,
        *,
        circuit: Circuit,
        front_points: Sequence[CircuitPoint],
        execute_list: Sequence[CircuitPoint],
        action: str,
        step_index: int,
        best_move: tuple[int, int] | None,
        brute_force_gate: tuple[int, ...] | None,
    ) -> None:
        if not self._forward_probe_matches(step_index):
            return
        summary = self._build_forward_probe_summary(
            circuit=circuit,
            front_points=front_points,
            execute_list=execute_list,
            action=action,
            step_index=step_index,
            best_move=best_move,
            brute_force_gate=brute_force_gate,
        )
        self.last_forward_probe = copy.deepcopy(summary)
        print(f'{self._log_prefix}[probe] {summary}')
        if (
            self._forward_probe_verbose_enabled()
            and action == 'bruteforce'
            and isinstance(getattr(self, 'last_bruteforce_trace', None), dict)
        ):
            brute_force_trace = copy.deepcopy(self.last_bruteforce_trace)
            for key in (
                'leading_moves',
                'move_call_trace',
                'move_step_trace',
                'resolve_trace',
                'segment_check_trace',
                'segment_drain_trace',
                'final_assignment',
            ):
                if key in brute_force_trace:
                    print(f'{self._log_prefix}[probe-detail] {key}: {brute_force_trace[key]}')

    def _deep_trace_enabled(self) -> bool:
        return os.getenv('BQSKIT_QCCD_DEEP_TRACE', '').lower() in (
            '1', 'true', 'yes', 'on',
        )

    def _append_resolve_trace(self, entry: dict[str, object]) -> None:
        probe_trace_active = (
            self._forward_probe_enabled()
            and self._active_forward_probe_step_matches()
        )
        if not self._capture_forward_trace_enabled() and not probe_trace_active:
            return
        if not hasattr(self, 'last_bruteforce_trace') or self.last_bruteforce_trace is None:
            return
        traces = self.last_bruteforce_trace.setdefault('resolve_trace', [])
        traces.append(copy.deepcopy(entry))

    def _append_deep_trace(
        self,
        key: str,
        entry: dict[str, object],
    ) -> None:
        probe_trace_active = (
            self._forward_probe_enabled()
            and self._active_forward_probe_step_matches()
        )
        if not self._deep_trace_enabled() and not (
            probe_trace_active
            and (
                key == 'segment_drain_trace'
                or key == 'segment_check_trace'
                or (
                    self._forward_probe_verbose_enabled()
                    and key in ('move_call_trace', 'move_step_trace')
                )
            )
        ):
            return
        if not hasattr(self, 'last_bruteforce_trace') or self.last_bruteforce_trace is None:
            return
        traces = self.last_bruteforce_trace.setdefault(key, [])
        traces.append(copy.deepcopy(entry))

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
        ion_assignment: dict[int, int],
    ) -> tuple[int, ...]:
        occupied_positions = {int(pos) for pos in ion_assignment.values()}
        occupied: list[int] = []
        position_int = int(position)
        if position_int in occupied_positions:
            occupied.append(position_int)
        for layer in layers:
            for node in layer:
                node_int = int(node)
                if node_int in occupied_positions:
                    occupied.append(node_int)
        return tuple(occupied)

    def _congestion_rate_from_layers(
        self,
        position: int,
        layers: tuple[tuple[int, ...], ...],
        ion_assignment: dict[int, int],
    ) -> tuple[float, float]:
        occupied_positions = {int(pos) for pos in ion_assignment.values()}
        position_int = int(position)
        congestion_score = 1.0 if position_int in occupied_positions else 0.0

        layer_weight = 1.0
        num_neighbors = 0
        num_occupied_neighbors = 0
        seen_neighbors: set[int] = set()
        for layer in layers:
            for node in layer:
                node_int = int(node)
                if node_int in occupied_positions:
                    congestion_score += layer_weight
                if node_int not in seen_neighbors:
                    seen_neighbors.add(node_int)
                    num_neighbors += 1
                    if node_int in occupied_positions:
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
        ion_assignment: dict[int, int],
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
            self._congestion_signature(position, layers, ion_assignment),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        cached = self._congestion_rate_from_layers(position, layers, ion_assignment)
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
        pre_assignment: dict[int, int],
        post_assignment: dict[int, int],
        action: str,
        best_move: tuple[int, int] | None = None,
        brute_force_gate: tuple[int, ...] | None = None,
    ) -> None:
        capture_forward_trace = self._capture_forward_trace_enabled()
        probe_enabled = self._forward_probe_enabled()
        if not capture_forward_trace and not probe_enabled:
            return
        step_index = int(getattr(self, '_active_forward_trace_step_index', 0))
        self._active_forward_trace_step_index = step_index + 1
        self._emit_forward_probe(
            circuit=circuit,
            front_points=front_points,
            execute_list=execute_list,
            action=action,
            step_index=step_index,
            best_move=best_move,
            brute_force_gate=brute_force_gate,
        )
        if not capture_forward_trace:
            return
        sorted_front = self._sorted_points(front_points)
        brute_force_trace = None
        if action == 'bruteforce':
            brute_force_trace = copy.deepcopy(getattr(self, 'last_bruteforce_trace', None))
        self.last_forward_trace.append({
            'action': action,
            'front_points': [str(n) for n in sorted_front],
            'front_locations': self._format_locations(circuit, sorted_front),
            'front_executable': [
                (
                    tuple(int(q) for q in circuit[n].location),
                    self.qccd_machine.gate_is_executable(circuit[n], self.pi, pre_assignment),
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
            pi: list[int],
            ion_assignment: dict,
            modify_circuit: bool
    ) -> None:
        """
        Apply a forward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                            maps logical qudits to physical qudits. So, `pi[l] == p`
                            implies logical qudit `l` is sitting on physical qudit `p`.

            ion_assignment (dict): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.

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
        decay = [1.0 for _ in range(self.qccd_machine.position_graph.num_qudits)]
        self.distance_stats = {
            'calls': 0,
            'single_qudit_calls': 0,
            'multi_qudit_calls': 0,
        }
        repeated_path = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        self.last_forward_trace: list[dict[str, object]] = []
        self.last_forward_probe = None
        self._active_forward_pass_index = (
            int(getattr(self, '_forward_pass_counter', 0)) + 1
        )
        self._forward_pass_counter = self._active_forward_pass_index
        self._active_forward_trace_step_index = 0
        self.pi = pi
        _logger.debug(
            '%s Starting forward sabre pass with ion assignment: %s.',
            self._log_prefix,
            ion_assignment,
        )
        # print(f"Starting forward sabre pass with ion assignment: {ion_assignment}.")

        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

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
                log_assignment = dict(ion_assignment)
                brute_force_moves = self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pi, ion_assignment,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_assignment,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(circuit.num_qudits),
                            list(range(circuit.num_qudits)),
                        )
                leading_moves += brute_force_moves
            # print("Current ion mapping: ", ion_assignment)
            current_front = self._sorted_points(F)
            pre_assignment = dict(ion_assignment)
            execute_list = self._sorted_points(
                [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], pi, ion_assignment)],
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
                    prev_executed_counts.pop(n)
                    _logger.debug('%s Executing gate at point %s.', self._log_prefix, n)
                    self._status(f'Executing gate at point {n}.')
                    if modify_circuit:
                        op = circuit[n]
                        physical_location = [pi[q] for q in op.location]
                        mapped_circuit.append_gate(op.gate, op.location)
                        instructions_list.append([f"Execute at {physical_location}", f"{ion_assignment}"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)
                self._record_forward_trace(
                    circuit=circuit,
                    front_points=current_front,
                    execute_list=execute_list,
                    pre_assignment=pre_assignment,
                    post_assignment=ion_assignment,
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
                        log_assignment = dict(ion_assignment)
                        brute_force_moves = self._brute_force_congestion(
                            circuit[self._sorted_points(F)[0]], D, pi, ion_assignment,
                        )
                        if modify_circuit:
                            for move in brute_force_moves:
                                if not self._append_logged_move_instruction(
                                    instructions_list,
                                    move,
                                    log_assignment,
                                    D,
                                ):
                                    continue
                                mapped_circuit.append_gate(
                                    BarrierPlaceholder(circuit.num_qudits),
                                    list(range(circuit.num_qudits)),
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
                    pi,
                    ion_assignment,
                    decay,
                    heuristic_move,
                )
            if best_move is None:
                brute_force_gate_point = self._sorted_points(F)[0]
                brute_force_gate = tuple(
                    int(q) for q in circuit[brute_force_gate_point].location
                )
                log_assignment = dict(ion_assignment)
                brute_force_moves = self._brute_force_congestion(
                    circuit[brute_force_gate_point], D, pi, ion_assignment,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_assignment,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(circuit.num_qudits),
                            list(range(circuit.num_qudits)),
                        )
                leading_moves += brute_force_moves
                self._record_forward_trace(
                    circuit=circuit,
                    front_points=current_front,
                    execute_list=execute_list,
                    pre_assignment=pre_assignment,
                    post_assignment=ion_assignment,
                    action='bruteforce',
                    brute_force_gate=brute_force_gate,
                )
                continue
            self._status(f'Best move: {best_move}')
            log_assignment = dict(ion_assignment)
            self._apply_move(best_move, ion_assignment)
            leading_moves.append(best_move)
            self._record_forward_trace(
                circuit=circuit,
                front_points=current_front,
                execute_list=execute_list,
                pre_assignment=pre_assignment,
                post_assignment=ion_assignment,
                action='move',
                best_move=best_move,
            )

            if modify_circuit:
                if self._append_logged_move_instruction(
                    instructions_list,
                    best_move,
                    log_assignment,
                    D,
                ):
                    mapped_circuit.append_gate(
                        BarrierPlaceholder(circuit.num_qudits),
                        list(range(circuit.num_qudits)),
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
            pi: list,
            ion_assignment: dict,
    ) -> list[tuple[int, int]]:
        """
            Logical function
        """
        gate_pos = []
        leading_moves = []
        congestion_cache: dict[tuple[object, ...], tuple[float, float]] = {}
        for p in gate.location:
            gate_pos.append(ion_assignment[pi[p]])
        initial_gate_pos = list(gate_pos)
        self._status(f'Trying to solve brute-force congestion at gate {gate_pos} with {ion_assignment}')
        #raise ValueError("Stopping for debug....")
        _logger.debug(
            '%s Trying to solve brute-force congestion at gate %s with %s',
            self._log_prefix,
            gate_pos,
            ion_assignment,
        )
        selected_trap_space = []
        selected_end_point = []
        relative_distance = np.inf
        selected_trap_id = None
        trap_candidates: list[dict[str, object]] = []
        # Select which trap to brute force in
        for trap in self.qccd_machine.physical_graph.executable_trap_list:
            all_trap_space = list(self.qccd_machine.physical_to_position[trap.id])
            # endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
            _, available_trap_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
            # Change to endpoints of trap space ... TODO
            raw_relative_dis_to_trap = self._get_distance_from_position_to_trap(gate_pos,
                                                                                all_trap_space,
                                                                                D,
                                                                                ion_assignment)
            """
                Only need to calculate unoccupied spaces (the one near the endpoints) if exits or only the endpoints...  
            """
            num_available_trap_space = len(available_trap_space)
            relative_dis_to_trap = (
                raw_relative_dis_to_trap - num_available_trap_space * 120e-6
            )
            trap_candidates.append({
                'trap_id': trap.id,
                'trap_space': [int(x) for x in all_trap_space],
                'available_space': [int(x) for x in available_trap_space],
                'available_count': int(num_available_trap_space),
                'raw_distance': float(raw_relative_dis_to_trap),
                'adjusted_distance': float(relative_dis_to_trap),
            })
            # print(f"Considering trap: {trap.id} with distance {relative_dis_to_trap} and number of available space :{num_available_trap_space}")
            if relative_dis_to_trap < relative_distance:
                selected_trap_space = all_trap_space
                # ToDo: If there are more than two endpoints?
                # selected_end_point = self.qccd_machine.trap_end_points[trap.id]
                selected_trap_id = trap.id
                relative_distance = relative_dis_to_trap
        for trap_candidate in trap_candidates:
            trap_candidate['selected'] = (
                trap_candidate['trap_id'] == selected_trap_id
            )
        selected_trap_space_unsorted = list(selected_trap_space)
        # print(f"Selected trap: {selected_trap_space}", )
        # Select the order of moving position
        distance_to_trap_lst = []
        for pos in gate_pos:
            tmp_distance_to_trap = [D[pos][trap_space] for trap_space in selected_trap_space]
            self._debug_compare(f'trap distance options for {pos}: {tmp_distance_to_trap}')
            distance_to_trap = float(np.min(tmp_distance_to_trap))
            end_point = selected_trap_space[int(np.argmax(tmp_distance_to_trap))]
            if end_point not in self.qccd_machine.trap_end_points[selected_trap_id]:
                end_point = self.qccd_machine.trap_end_points[selected_trap_id][0]
            selected_end_point.append(end_point)
            distance_to_trap_lst.append(distance_to_trap)
        gate_pos = np.array(gate_pos)[np.argsort(distance_to_trap_lst)]
        selected_end_point = np.array(selected_end_point)[np.argsort(distance_to_trap_lst)]
        ion_order = [list(ion_assignment.keys())[list(ion_assignment.values()).index(i)] for i in gate_pos]
        _logger.debug(f"Selected end point: {selected_end_point}", )
        # print(f"Selected end point: {selected_end_point}")
        # print("Order of moving ions: ", ion_order)
        for ion_index in range(len(ion_order)):
            gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
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
            'trap_candidates': trap_candidates,
            'selected_trap_space_initial': [int(x) for x in selected_trap_space_unsorted],
            'selected_trap_space_sorted': [int(x) for x in list(selected_trap_space)],
            'selected_end_point': [int(x) for x in list(selected_end_point)],
            'distance_to_trap_lst': [float(x) for x in distance_to_trap_lst],
            'ion_order': [int(x) for x in ion_order],
            'leading_moves': [],
        }
        if self._backward_target_gate_matches(gate.location):
            self._emit_backward_bruteforce_trap_probe(
                {
                    'backward_pass': int(
                        getattr(self, '_active_backward_pass_index', -1),
                    ),
                    'gate_location': tuple(int(q) for q in gate.location),
                    'initial_gate_pos': [int(x) for x in initial_gate_pos],
                    'trap_candidates': self._snapshot_trace_value(trap_candidates),
                    'selected_trap_id': selected_trap_id,
                },
            )
        # print("Order selected traps: ", selected_trap_space)

        # Move the pos to the selected trap
        for pos_idx in range(len(gate_pos)):
            move_source = int(gate_pos[pos_idx])
            move_target = int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx])
            self._append_deep_trace('move_call_trace', {
                'phase': 'gate_move',
                'pos_idx': int(pos_idx),
                'source': move_source,
                'target': move_target,
                'clearing_ep': False,
                'assignment_before': dict(ion_assignment),
            })
            self._status(
                f'Trying to moving ion {gate_pos[pos_idx]}... to '
                f'{move_target}',
            )
            #_logger.debug(f"Trying to moving ion {gate_pos[pos_idx]}...")
            # if pos_idx != len(gate_pos) - 1:
            #     print(f"Endpoint: {selected_end_point[pos_idx+1]}")
            leading_moves += self._brute_force_move(
                int(gate_pos[pos_idx]),
                int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx]),
                ion_assignment,
                congestion_cache=congestion_cache,
            )
            self.last_bruteforce_trace['leading_moves'] = [
                tuple(int(v) for v in move) for move in leading_moves
            ]
            self._status(f'Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}')
            # _logger.debug(f"Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}")
            # _logger.debug(f"Selected end point: {selected_end_point}")
            gate_pi = [pi[p] for p in gate.location]
            # _logger.debug(f"Gate pi: {gate_pi}")
            # if selected_end_point in ion_assignment.values():
            #     _logger.debug(f"Position of ion: {list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point)]}")
            """
                If too many ions are in the segment, move them back to trap. (Disable when trying H2 architecture)
            """
            number_of_segment = len(self.qccd_machine.physical_to_position["segment_space"])
            segment_space = [
                int(position)
                for position in self.qccd_machine.physical_to_position["segment_space"]
            ]
            ion_at_segment = []
            for ion in ion_assignment.keys():
                if ion_assignment[ion] in self.qccd_machine.physical_to_position["segment_space"]:
                    ion_at_segment.append(ion)
            segment_occupancy_count = len(ion_at_segment)
            segment_occupancy_ratio = (
                float(segment_occupancy_count / number_of_segment)
                if number_of_segment != 0 else 0.0
            )
            trap_status = []
            for trap in self.qccd_machine.physical_graph.trap_list:
                trap_positions = [
                    int(position)
                    for position in self.qccd_machine.physical_to_position[trap.id]
                ]
                occupied_positions = [
                    int(ion_assignment[ion])
                    for ion in sorted(ion_assignment.keys())
                    if ion_assignment[ion] in trap_positions
                ]
                is_full, available_space = self.qccd_machine.trap_is_fully_occupied(
                    trap.id,
                    ion_assignment,
                )
                trap_status.append({
                    'trap_id': trap.id,
                    'capacity': int(trap.max_num_ions),
                    'occupied_count': len(occupied_positions),
                    'occupied_positions': occupied_positions,
                    'available_count': len(available_space),
                    'available_spaces': [int(space) for space in available_space],
                    'is_full': bool(is_full),
                })
            self._append_deep_trace('segment_check_trace', {
                'phase': 'pre_check',
                'pos_idx': int(pos_idx),
                'segment_space': segment_space,
                'number_of_segment': int(number_of_segment),
                'segment_occupancy_count': int(segment_occupancy_count),
                'segment_occupancy_ratio': segment_occupancy_ratio,
                'congestion_rate_threshold': float(self.cogestion_segment_rate),
                'drain_will_trigger': (
                    segment_occupancy_ratio >= float(self.cogestion_segment_rate)
                ),
                'logical_positions': {
                    int(ion): int(ion_assignment[ion])
                    for ion in sorted(ion_assignment.keys())
                },
                'segment_members': [
                    {
                        'logical': int(ion),
                        'position': int(ion_assignment[ion]),
                    }
                    for ion in ion_at_segment
                ],
                'trap_status': trap_status,
            })
            if len(ion_at_segment) / number_of_segment >= self.cogestion_segment_rate:
                self._status('As there are many ions outside the traps, move them to the trap...')
                available_spaces = []
                for trap in self.qccd_machine.physical_graph.trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
                    available_spaces += available_space
                self._append_deep_trace('segment_drain_trace', {
                    'phase': 'trigger',
                    'pos_idx': int(pos_idx),
                    'ion_at_segment': [int(ion) for ion in ion_at_segment],
                    'available_spaces': [int(space) for space in available_spaces],
                    'assignment_before': dict(ion_assignment),
                })
                while ion_at_segment:
                    drain_source = int(ion_assignment[ion_at_segment[0]])
                    drain_target = int(available_spaces[0])
                    self._append_deep_trace('move_call_trace', {
                        'phase': 'segment_drain',
                        'pos_idx': int(pos_idx),
                        'source': drain_source,
                        'target': drain_target,
                        'logical': int(ion_at_segment[0]),
                        'clearing_ep': False,
                        'assignment_before': dict(ion_assignment),
                    })
                    leading_moves += self._brute_force_move(
                        drain_source,
                        drain_target,
                        ion_assignment,
                        congestion_cache=congestion_cache,
                    )
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                    available_spaces = []
                    for trap in self.qccd_machine.physical_graph.trap_list:
                        _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
                        available_spaces += available_space
                    ion_at_segment = []
                    for ion in ion_assignment.keys():
                        if ion_assignment[ion] in self.qccd_machine.physical_to_position["segment_space"]:
                            ion_at_segment.append(ion)
                    self._append_deep_trace('segment_drain_trace', {
                        'phase': 'after_iteration',
                        'pos_idx': int(pos_idx),
                        'ion_at_segment': [int(ion) for ion in ion_at_segment],
                        'available_spaces': [int(space) for space in available_spaces],
                        'assignment_after': dict(ion_assignment),
                    })
                    # print("Ion at segment: ", ion_at_segment)
                    # print("Available trap space: ", available_spaces)
                    # print(f"Trying to solve brute-force congestion at gate {gate_pos} with {ion_assignment}")
                for ion_index in range(len(ion_order)):
                    gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
            """
                Clearing the end-point of the selected trap
            """
            if pos_idx == len(gate_pos) - 1:
                continue
            elif (selected_end_point[pos_idx + 1] in ion_assignment.values() and
                  (pos_idx != len(gate_pos) - 1 and
                   list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point[pos_idx + 1])]
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
                    if neighbor in list(ion_assignment.values()):
                        occupied_neighbors.append(neighbor)
                end_point_neighbors = [i for i in end_point_neighbors if i not in occupied_neighbors]
                if not end_point_neighbors:
                    potential_blockage = [i for i in occupied_neighbors if self.qccd_machine.get_trap_id(i) is None]
                    self._append_deep_trace('move_call_trace', {
                        'phase': 'clear_endpoint',
                        'pos_idx': int(pos_idx),
                        'source': int(selected_end_point[pos_idx + 1]),
                        'target': int(potential_blockage[0]),
                        'clearing_ep': True,
                        'assignment_before': dict(ion_assignment),
                    })
                    leading_moves += self._brute_force_move(
                        int(selected_end_point[pos_idx + 1]),
                        int(potential_blockage[0]),
                        ion_assignment,
                        clearing_ep=True,
                        congestion_cache=congestion_cache,
                    )
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                else:
                    self._apply_move((selected_end_point[pos_idx + 1], end_point_neighbors[0]), ion_assignment)
                    leading_moves.append(tuple(sorted((selected_end_point[pos_idx + 1], end_point_neighbors[0]))))
                    self.last_bruteforce_trace['leading_moves'] = [
                        tuple(int(v) for v in move) for move in leading_moves
                    ]
                    # print(f"Perform move {(selected_end_point[pos_idx + 1], end_point_neighbors[0])} to clear "
                    #       f"the endpoint")
            for ion_index in range(len(ion_order)):
                gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
            self._status(f'Gate position gate updated to {gate_pos}--------------c')
        self.last_bruteforce_trace['final_assignment'] = dict(ion_assignment)
        return leading_moves

    def _brute_force_move(
            self,
            position: int,
            trap_space: int,
            ion_assignment: dict,
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
        self._append_deep_trace('move_step_trace', {
            'phase': 'move_start',
            'source': int(position),
            'target': int(trap_space),
            'path': [int(p) for p in path],
            'clearing_ep': bool(clearing_ep),
            'initial_ion_status': str(ion_status),
            'assignment_before': dict(ion_assignment),
        })
        for idx_point in range(len(path) - 1):
            possible_move = (path[idx_point], path[idx_point + 1])
            step_entry = {
                'phase': 'move_step',
                'step_index': int(idx_point),
                'source': int(path[idx_point]),
                'target': int(path[idx_point + 1]),
                'ion_status_before': str(ion_status),
                'next_occupied': bool(path[idx_point + 1] in ion_assignment.values()),
            }
            if path[idx_point + 1] not in ion_assignment.values():
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                step_entry['decision'] = 'empty_neighbor'
                # print(
                #     f"Perform move {(possible_move, ion_assignment)} as there is no ion in the neighbor, "
                #     f"ion status: {ion_status}")
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                step_entry['decision'] = 'trap_inner_swap'
                # print(f"Perform move {possible_move} with inner-swap, ion status: {ion_status}")
            else:
                ion_pos = path[idx_point]
                blockage = path[idx_point + 1]
                step_entry['decision'] = 'resolve_congestion'
                step_entry['blockage'] = int(blockage)
                # print(f"There is blockage at {blockage}, try to resolve it...")
                leading_moves += self._resolve_congestion(
                    ion_pos,
                    path,
                    blockage,
                    ion_assignment,
                    ion_pos,
                    blockage,
                    congestion_cache=congestion_cache,
                )
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                # print(f"Perform move {possible_move} after resolving blockage")
                # print(f"Ion assignment after resolving blockage: {ion_assignment}")
            if ion_status == 'segment' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                ion_status = 'trap'
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'segment':
                ion_status = 'segment'
            step_entry['ion_status_after'] = str(ion_status)
            step_entry['assignment_after'] = dict(ion_assignment)
            self._append_deep_trace('move_step_trace', step_entry)
        return leading_moves

    def _resolve_congestion(
            self,
            target: int,
            path: list[int],
            blockage: int,
            ion_assignment: dict,
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
        # print(
        #     f"Trying to resolve blockage {blockage} from the target {target} path with ion assignment {ion_assignment}")
        _logger.debug(
            f"Trying to resolve blockage {blockage} from the target {target} path with ion assignment {ion_assignment}")
        leading_moves = []
        # print("Path: {}".format(path))
        # print("Original target: ", original_target)
        # print("Original blockage: ", original_blockage)
        target_ion_index = list(ion_assignment.keys())[list(ion_assignment.values()).index(target)]
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
                    if neighbor_ext not in ion_assignment.values() and neighbor_ext not in path:
                        removed_labeled = False
                if removed_labeled:
                    removed_blockage_neighbors.append(neighbor)

        blockage_neighbors = [i for i in blockage_neighbors if i not in removed_blockage_neighbors]
        # print("Blockage_neighbors: ", blockage_neighbors)
        # _logger.debug(f"Blockage neighbors: {blockage_neighbors}")
        potential_blockage = []
        for neighbor in blockage_neighbors:
            if neighbor in ion_assignment.values():
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
            if congestion_cache is None:
                congestion_cache = {}
            congestion = np.array([
                self._cached_congestion_rate(
                    blockage_neighbor,
                    target,
                    blockage,
                    ion_assignment,
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
            self._apply_move((blockage, blockage_neighbors[choosen_idx]), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, blockage_neighbors[choosen_idx]))))
            # print(f"Blockage: {blockage}, blockage neighbors: {blockage_neighbors[choosen_idx]}")
            # print(
            #     f"Perform move (1) {(blockage, blockage_neighbors[choosen_idx])} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            return leading_moves
        elif potential_blockage:
            if congestion_cache is None:
                congestion_cache = {}
            congestion = np.array([
                self._cached_congestion_rate(
                    blockage_neighbor,
                    target,
                    blockage,
                    ion_assignment,
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
                    ion_assignment,
                    depth=self.qccd_machine.max_ion_capacity - 1 + num_call,
                    cache=congestion_cache,
                )[0] <= congestion_rates[choosen_idx]:
                    # Reverse move (treat target as blockage and vice versa)
                    # print(f"Blockage: {blockage}, target: {target}")
                    leading_moves += self._resolve_congestion(blockage, [], target, ion_assignment,
                                                                    original_target, original_blockage, num_call + 1,
                                                                    congestion_cache=congestion_cache)
                    self._apply_move((blockage, target), ion_assignment)
                    leading_moves.append(tuple(sorted((blockage, target))))
                    # print(
                    #     f"Perform move (2) {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                    blockage = target
                    target = ion_assignment[target_ion_index]
                    # print(f"Blockage: {blockage}, target: {target}")
                    if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                            and self.qccd_machine.get_trap_id(blockage) is None):
                        leading_moves += self._resolve_congestion(target, [], blockage, ion_assignment,
                                                                  original_target, original_blockage, num_call+1,
                                                                  congestion_cache=congestion_cache)
                    self._apply_move((blockage, target), ion_assignment)
                    leading_moves.append(tuple(sorted((blockage, target))))
                    # print(
                    #     f"Perform move (2') {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                else:
                    raise ValueError("This method does not resolve this case !!!")
            else:
                self._append_resolve_trace(resolve_entry)
                #print(f"Choose to resolve {potential_blockage[choosen_idx]}")
                leading_moves += self._resolve_congestion(blockage, path, potential_blockage[choosen_idx],
                                                          ion_assignment, original_target, original_blockage,
                                                          num_call + 1,
                                                          congestion_cache=congestion_cache)
                self._apply_move((blockage, potential_blockage[choosen_idx]), ion_assignment)
                leading_moves.append(tuple(sorted((blockage, potential_blockage[choosen_idx]))))
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
            leading_moves += self._resolve_congestion(blockage, [], target, ion_assignment,
                                                      original_target, original_blockage, num_call+1,
                                                      congestion_cache=congestion_cache)
            self._apply_move((blockage, target), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, target))))
            # print(
            #     f"Perform move (4) {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            blockage = target
            target = ion_assignment[target_ion_index]
            # print(f"Blockage: {blockage}, target: {target}")
            if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                    and self.qccd_machine.get_trap_id(blockage) is None):
                leading_moves += self._resolve_congestion(target, [], blockage, ion_assignment,
                                                          original_target, original_blockage, num_call+1,
                                                          congestion_cache=congestion_cache)
            self._apply_move((blockage, target), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, target))))
            # print(
            #     f"Perform move (4') {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
        return leading_moves

    def congestion_rate(self,
                        position: int,
                        target: int,
                        blockage: int,
                        ion_assignment: dict,
                        depth: int = 2):
        cache: dict[tuple[object, ...], tuple[float, float]] = {}
        return self._cached_congestion_rate(
            position,
            target,
            blockage,
            ion_assignment,
            depth=depth,
            cache=cache,
        )

    ######################################################################################
    def backward_pass(
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict
    ) -> None:
        """
        Apply a backward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                            maps logical qudits to physical qudits. So, `pi[l] == p`
                            implies logical qudit `l` is sitting on physical qudit `p`.

            ion_assignment (dict): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.
        """
        # Preprocessing
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.rear
        decay = [1.0 for _ in range(self.qccd_machine.position_graph.num_qudits)]
        self.distance_stats = {
            'calls': 0,
            'single_qudit_calls': 0,
            'multi_qudit_calls': 0,
        }
        repeated_path = False
        executed_flag = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        leading_moves: list[tuple[int, int]] = []
        heuristic_move = True
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        self.last_backward_trace = []
        self._last_backward_move_search = None
        self._active_backward_pass_index = (
            int(getattr(self, '_backward_pass_counter', 0)) + 1
        )
        self._backward_pass_counter = self._active_backward_pass_index
        _logger.debug(
            '%s Starting backward sabre QCCD pass with ion assignment: %s.',
            self._log_prefix,
            pi,
        )
        longest_path = self.qccd_machine.get_longest_move_path_length()
        # Main Loop
        while len(F) > 0:
            execute_candidates = self._sorted_points(
                [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], pi, ion_assignment)],
            )
            front_locations = self._format_locations(circuit, self._sorted_points(F))
            execute_locations = self._format_locations(circuit, execute_candidates)
            pre_assignment = dict(sorted(ion_assignment.items()))
            self._emit_backward_probe(
                'loop',
                {
                    'backward_pass': int(getattr(self, '_active_backward_pass_index', -1)),
                    'iter_count': int(self.iter_count),
                    'front_locations': front_locations,
                    'execute_locations': execute_locations,
                    'assignment': pre_assignment,
                },
            )
            # Retrieve executable gates giving the current ion assignment: pi
            # print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                # print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path/4):
                # print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                self._emit_backward_probe(
                    'bruteforce-trigger',
                    {
                        'iter_count': int(self.iter_count),
                        'front_locations': front_locations,
                    },
                )
                self._record_backward_trace(
                    {
                        'action': 'bruteforce',
                        'front_locations': front_locations,
                        'execute_locations': execute_locations,
                        'best_move': None,
                        'move_search': None,
                        'pre_assignment': pre_assignment,
                        'post_assignment': None,
                    },
                )
                leading_moves += self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pi, ion_assignment,
                )
            # print("Current ion mapping: ", ion_assignment)
            execute_list = execute_candidates
            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                execute_probe_active = self._backward_execute_probe_matches(
                    circuit,
                    execute_list,
                )
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
                post_assignment = dict(sorted(ion_assignment.items()))
                if execute_probe_active:
                    self._emit_backward_execute_probe(
                        {
                            'backward_pass': int(getattr(self, '_active_backward_pass_index', -1)),
                            'front_locations': front_locations,
                            'execute_locations': execute_locations,
                            'pre_assignment': pre_assignment,
                            'post_assignment': post_assignment,
                            'assignment_delta': self._assignment_delta(
                                pre_assignment,
                                post_assignment,
                            ),
                            'remaining_front_locations': self._format_locations(
                                circuit,
                                self._sorted_points(F),
                            ),
                        },
                    )
                self._record_backward_trace(
                    {
                        'action': 'execute',
                        'front_locations': front_locations,
                        'execute_locations': execute_locations,
                        'best_move': None,
                        'move_search': None,
                        'pre_assignment': pre_assignment,
                        'post_assignment': post_assignment,
                    },
                )
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
                            circuit[self._sorted_points(F)[0]], D, pi, ion_assignment,
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
                    circuit,
                    F,
                    E,
                    D,
                    pi,
                    ion_assignment,
                    decay,
                    heuristic_move,
                )
            if best_move is None:
                brute_force_probe_active = self._backward_gate_probe_matches_front(
                    circuit,
                    self._sorted_points(F),
                )
                self._emit_backward_probe(
                    'best-move',
                    {
                        'result': None,
                        'front_locations': front_locations,
                    },
                )
                self._record_backward_trace(
                    {
                        'action': 'no-move',
                        'front_locations': front_locations,
                        'execute_locations': execute_locations,
                        'best_move': None,
                        'move_search': self._snapshot_trace_value(
                            getattr(self, '_last_backward_move_search', None),
                        ),
                        'pre_assignment': pre_assignment,
                        'post_assignment': None,
                    },
                )
                leading_moves += self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pi, ion_assignment,
                )
                if brute_force_probe_active:
                    post_assignment = dict(sorted(ion_assignment.items()))
                    brute_force_trace = self._snapshot_trace_value(
                        getattr(self, 'last_bruteforce_trace', None),
                    )
                    self._emit_backward_bruteforce_probe(
                        {
                            'backward_pass': int(getattr(self, '_active_backward_pass_index', -1)),
                            'front_locations': front_locations,
                            'execute_locations': execute_locations,
                            'pre_assignment': pre_assignment,
                            'post_assignment': post_assignment,
                            'assignment_delta': self._assignment_delta(
                                pre_assignment,
                                post_assignment,
                            ),
                            'bruteforce_trace': brute_force_trace,
                        },
                    )
                continue
            # print(f"Best move: {best_move}")
            self._apply_move(best_move, ion_assignment)
            leading_moves.append(best_move)
            self.iter_count += 1
            post_assignment = dict(sorted(ion_assignment.items()))
            self._emit_backward_probe(
                'applied-move',
                {
                    'move': tuple(int(x) for x in best_move),
                    'iter_count_after': int(self.iter_count),
                    'assignment_after': post_assignment,
                },
            )
            self._record_backward_trace(
                {
                    'action': 'move',
                    'front_locations': front_locations,
                    'execute_locations': execute_locations,
                    'best_move': tuple(int(x) for x in best_move),
                    'move_search': self._snapshot_trace_value(
                        getattr(self, '_last_backward_move_search', None),
                    ),
                    'pre_assignment': pre_assignment,
                    'post_assignment': post_assignment,
                },
            )

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
            pi: list,
            ion_assignment: dict,
            decay: list[float],
            heuristic_move: bool,
    ) -> tuple[int, int]:
        """Return the best move given the current algorithm state and ion assignment. (Logical function)"""
        # Track best one
        best_score = np.inf
        best_move = None

        # Gather all considerable moves
        if heuristic_move:
            move_candidate_list = self._obtain_heuristic_moves(circuit, F, pi, ion_assignment)
        else:
            move_candidate_list = self._obtain_moves(circuit, pi, ion_assignment)
        frontier_points = self._sorted_points(F)
        frontier_locations = [tuple(int(q) for q in circuit[n].location) for n in frontier_points]
        self._debug_compare(f'frontier points={frontier_points} logical_locations={frontier_locations}')
        self._debug_compare(f'current assignment={dict(sorted(ion_assignment.items()))}')
        self._debug_compare(f'candidate moves={self._sorted_moves(move_candidate_list)}')
        self._emit_backward_probe(
            'move-search',
            {
                'front_locations': frontier_locations,
                'candidate_moves': self._sorted_moves(move_candidate_list),
                'extended_locations': self._format_locations(circuit, self._sorted_points(E)),
            },
        )
        list_of_best_move = []
        move_scores = []
        # Score them, tracking the best one
        # scores = Parallel(n_jobs=5)(delayed(self._score_move)(circuit, F, D, pi, ion_assignment, move, decay, E)
        #                             for move in move_candidate_list)
        # list_of_best_score = np.argwhere(scores == np.max(scores)).flatten().tolist()
        # list_of_best_moves = list(move_candidate_list)[list_of_best_score]
        for move in self._sorted_moves(move_candidate_list):
            score = self._score_move(circuit, F, D, pi, ion_assignment, move, decay, E)
            move_scores.append((move, float(score)))
            if score < best_score:
                best_score = score
                best_move = move
                list_of_best_move = [move]
            elif score == best_score:
                list_of_best_move.append(move)
        self._debug_compare(f'move scores={move_scores}')
        self._emit_backward_probe(
            'move-scores',
            {
                'scores': move_scores,
                'best_score': None if best_move is None else float(best_score),
                'best_moves': list_of_best_move,
            },
        )
        self._last_backward_move_search = {
            'front_locations': frontier_locations,
            'candidate_moves': self._sorted_moves(move_candidate_list),
            'extended_locations': self._format_locations(circuit, self._sorted_points(E)),
            'scores': move_scores,
            'best_score': None if best_move is None else float(best_score),
            'best_moves': [
                tuple(int(x) for x in move)
                for move in list_of_best_move
            ],
        }
        if best_move is None:
            # print("*** Unable to find best move. ***")
            return None
            # raise RuntimeError('Unable to find best move.')
        # print(f"List of best move: {list_of_best_move}")
        if len(list_of_best_move) == 1:
            self._debug_compare(f'chosen move={best_move} best_score={float(best_score)}')
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
                    p = [ion_assignment[pi[loc]] for loc in location]
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
            pi: list,
            ion_assignment: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves w.r.t frontier given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = []
        for location in list(F)[:1]:
            block = circuit[location]
            for qudit in block.location:
                physical_qudit_positions.append(ion_assignment[pi[qudit]])
        physical_qudit_positions = self._sorted_unique_ints(physical_qudit_positions)
        moves = set()
        for physical_qudit_position in physical_qudit_positions:
            neighbors = sorted(position_graph.get_neighbors_of(physical_qudit_position))
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if a in list(ion_assignment.values()) and b in list(ion_assignment.values()):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _obtain_moves(
            self,
            circuit: Circuit,
            pi: list,
            ion_assignment: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = self._sorted_unique_ints(
            ion_assignment[pi[i]] for i in circuit.active_qudits
        )
        moves = set()
        for physical_qudit_position in physical_qudit_positions:
            neighbors = sorted(position_graph.get_neighbors_of(physical_qudit_position))
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if a in list(ion_assignment.values()) and b in list(ion_assignment.values()):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _score_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            D: list[list[float]],
            pi: list,
            ion_assignment: dict,
            move: tuple[int, int],
            decay: list[float],
            E: set[CircuitPoint]
    ) -> float:
        """Score the candidate realizable physical moves given the current algorithm state and ion assignment."""
        # Apply potential move  which is physical
        # print("Initial ion assignement: ", pi)
        l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
            if move[0] in list(ion_assignment.values()) else None
        l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
            if move[1] in list(ion_assignment.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')

        if l1 is None:
            ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[1], move[0]  # Inner trap swap
        # print("Ion assignment after moved: ", pi)
        # Calculate front set term
        front = 0.0
        for n in list(F):
            logical_qudits = circuit[n].location
            front += self._get_distance(logical_qudits, pi, ion_assignment, D)
        front /= len(F)

        # Calculate extended set term
        extend = 0.0
        # if len(E) > 0:
        #     for n in E:
        #         extend += self._get_distance(circuit[n].location, pi, ion_assignment, D)
        #     extend /= len(E)
        #     extend *= self.extended_set_weight

        # Calculate decay factor
        # decay_factor = max(decay[move[0]], decay[move[1]])
        # Undo potential move
        if l1 is None:
            ion_assignment[l2] = move[1]  # Re-move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[0]  # Re-move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[0], move[1]  # Inner trap swap
        # print(f"Calculating score move {move} w.r.t pi: {pi} yields the front value of {front}"
        #       f" and extend value of {extend}")
        # print("-------------------------------------------------------------------------------")
        return front + extend

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
            ion_assignment: dict
    ) -> float:
        """
            Get minimum distance from all the position of the gate to the trap...
        """
        distance = np.inf
        """
            All position of trap are considered...
        """
        #print("Available space: ", available_space)
        for space in permutations(available_space, len(positions)):
            #print("Considering space combination: {}".format(space))
            blockage = [self.qccd_machine.path_is_blocked(positions[i], space[i], ion_assignment)[1]
                        for i in range(len(positions))]
            #print("Blockage: ", blockage)
            space_distance = np.sum([D[positions[i]][space[i]] for i in range(len(positions))])
            #print("Space distance: ", space_distance)
            for block_w in blockage:
                for block_position in block_w:
                    if self.qccd_machine.position_to_physical[block_position] == 'segment':
                        resolve_cost = self.qccd_machine.timing_data['junction_Y']
                    elif self.qccd_machine.position_to_physical[block_position] == 'trap':
                        min_to_endpoints = np.min([D[block_position][end_point] for end_point in
                                                   self.qccd_machine.trap_end_points[
                                                       self.qccd_machine.get_trap_id(block_position)]])
                        resolve_cost = (self.qccd_machine.timing_data['split'] + min_to_endpoints)
                    else:
                        raise ValueError("The block position is undefined as it sit on ",
                                         self.qccd_machine.position_to_physical[block_position])
                    space_distance += resolve_cost
            # print(f"Distance when considering space {space}: ", space_distance)
            # print(f"Current minimum distance: ", distance)
            # print("........")
            distance = np.min([space_distance, distance])
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
            pi: list,
            ion_assignment: dict,
            D: list[list[float]],
    ) -> float:
        """
            Calculate the expected cost w.r.t distance to connect logical qudits.
        """
        self.distance_stats['calls'] += 1
        # Single qudit case
        if len(logical_qudits) == 1:
            self.distance_stats['single_qudit_calls'] += 1
            p = [ion_assignment[pi[logical_qudits[0]]]]
            trap_p = self.qccd_machine.get_trap_id(p[0])
            if trap_p is not None:
                return 0.0
            else:
                distance_to_trap = np.inf
                for trap in self.qccd_machine.physical_graph.executable_trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
                    #endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
                    # Change to endpoints of trap space ... TODO
                    distance_to_trap = np.min([self._get_distance_from_position_to_trap(p, available_space, D,
                                                                                        ion_assignment),
                                               distance_to_trap])
                return distance_to_trap
        # Multi-qudit case
        self.distance_stats['multi_qudit_calls'] += 1
        p_list = [ion_assignment[pi[logical_qudit]] for logical_qudit in logical_qudits]
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
                _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
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
                    considering_dist_to_F = self._get_distance_from_position_to_trap(considering_p,
                                                                                     available_space,
                                                                                     D,
                                                                                     ion_assignment)
                total_F = np.min([total_F, considering_dist_to_F])
        # print(
        #     f"Physical distance w.r.t gate {logical_qudits} is {distance} "
        #     f"Total distance to nearest similar trap: {total_F}"
        # )
        return distance + total_F

    def _apply_move(
            self,
            move: tuple[int, int],
            ion_assignment: dict,
    ) -> None:
        """Apply the move to `pi` and update `decay`."""
        _logger.debug('applying move %s' % str(move))
        # Apply potential move
        l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
            if move[0] in list(ion_assignment.values()) else None
        l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
            if move[1] in list(ion_assignment.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
        if l1 is None:
            ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[1], move[0]  # Inner trap swap
        _logger.debug('ion assignment after move %s' % str(ion_assignment))
        # decay[move[0]] += self.decay_delta
        # decay[move[1]] += self.decay_delta

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
    mapping_algo.forward_pass(circuit, pi, ion_assignment, modify_circuit=True)
