from bqskit.ir import (CircuitPoint, CircuitRegion, CircuitPointLike,
                       CircuitRegionLike, Operation, CircuitLocation)
from bqskit.utils.typing import is_integer, _logger
from typing import Tuple, List, Set

def surround(
        self,
        point: CircuitPointLike,
        num_qudits: int,
        bounding_region: CircuitRegionLike | None = None,
        fail_quickly: bool = False,
) -> CircuitRegion:
    if not is_integer(num_qudits):
        raise TypeError(
            f'Expected an integer num_qudits, got {type(num_qudits)}.',
        )

    if num_qudits <= 0:
        raise ValueError(
            f'Expected a positive integer num_qudits, got {num_qudits}.',
        )

    if bounding_region is not None:
        bounding_region = CircuitRegion(bounding_region)

    point = self.normalize_point(point)

    init_op: Operation = self[point]  # Allow starting at an idle point

    if init_op.num_qudits > num_qudits:
        raise ValueError('Gate at point is too large for num_qudits.')

    HalfWire = Tuple[CircuitPoint, str]
    """
    A HalfWire is a point in the circuit and a direction.
    
    This represents a point to start exploring from and a direction to
    explore in.
    """

    Node = Tuple[
        List[HalfWire],
        Set[Tuple[int, Operation]],
        CircuitLocation,
        Set[CircuitPoint],
    ]
    """
    A Node in the search tree.
    
    Each node represents a region that may grow further. The data structure
    tracks all HalfWires in the region and the set of operations inside the
    region. During node exploration each HalfWire is walked until we find a
    multi-qudit gate. Multi- qudit gates form branches in the tree on
    whether on the gate should be included. The node structure additionally
    stores the set of qudit indices involved in the region currently. Also,
    we track points that have already been explored to reduce repetition.
    """

    # Initialize the frontier
    init_node = (
        [
            (CircuitPoint(point[0], qudit_index), 'left')
            for qudit_index in init_op.location
        ]
        + [
            (CircuitPoint(point[0], qudit_index), 'right')
            for qudit_index in init_op.location
        ],
        {(point[0], init_op)},
        init_op.location,
        {CircuitPoint(point[0], q) for q in init_op.location},
    )
    frontier: list[Node] = [init_node]

    # Track best so far
    def score(node: Node) -> int:
        return sum(op[1].num_qudits for op in node[1])

    best_score = score(init_node)
    best_region = self.get_region({(point[0], init_op.location[0])})

    # Exhaustive Search
    while len(frontier) > 0:
        node = frontier.pop(0)
        _logger.debug('popped node:')
        _logger.debug(node[0])
        _logger.debug(f'Items remaining in the frontier: {len(frontier)}')
        # Evaluate node
        if score(node) > best_score:
            # Calculate region from the best node and return
            points = {(cycle, op.location[0]) for cycle, op in node[1]}

            try:
                best_region = self.get_region(points)
                best_score = score(node)
                _logger.debug(f'new best: {best_region}.')

            # Need to reject bad regions
            except ValueError:
                if fail_quickly:
                    continue

        # Expand node
        absorbed_gates: set[tuple[int, Operation]] = set()
        branches: set[tuple[int, int, Operation]] = set()
        before_branch_half_wires: dict[int, HalfWire] = {}
        for i, half_wire in enumerate(node[0]):
            cycle_index, qudit_index = half_wire[0]
            step = -1 if half_wire[1] == 'left' else 1

            while True:

                # Take a step
                cycle_index += step

                # Stop at edges
                if cycle_index < 0 or cycle_index >= self.num_cycles:
                    break

                # Stop when outside bounds
                if bounding_region is not None:
                    if (cycle_index, qudit_index) not in bounding_region:
                        break

                # Stop when exploring previously explored points
                point = CircuitPoint(cycle_index, qudit_index)
                if point in node[3]:
                    break
                node[3].add(point)

                # Continue until next operation
                if self.is_point_idle(point):
                    continue
                op: Operation = self[cycle_index, qudit_index]

                # Gates already in region stop the half_wire
                if (cycle_index, op) in node[1]:
                    break

                # Gates already accounted for stop the half_wire
                if (cycle_index, op) in absorbed_gates:
                    break

                if (cycle_index, op) in [(c, o) for h, c, o in branches]:
                    break

                # Absorb single-qudit gates
                if len(op.location) == 1:
                    absorbed_gates.add((cycle_index, op))
                    continue

                # Operations that are too large stop the half_wire
                if len(op.location.union(node[2])) > num_qudits:
                    break

                # Otherwise branch on the operation
                branches.add((i, cycle_index, op))

                # Track state of half wire right before branch
                prev_point = CircuitPoint(cycle_index - step, qudit_index)
                before_branch_half_wires[i] = (prev_point, half_wire[1])
                break

            # Compute children and extend frontier
            for half_wire_index, cycle_index, op in branches:
                child_half_wires = [
                    half_wire
                    for i, half_wire in before_branch_half_wires.items()
                    if half_wire_index != i
                ]

                qudit = node[0][half_wire_index][0].qudit
                direction = node[0][half_wire_index][1]
                left_expansion = [
                    (CircuitPoint(cycle_index, qudit_index), 'left')
                    for qudit_index in op.location
                    if qudit != qudit_index or direction == 'left'
                ]
                right_expansion = [
                    (CircuitPoint(cycle_index, qudit_index), 'right')
                    for qudit_index in op.location
                    if qudit != qudit_index or direction == 'right'
                ]
                expansion = left_expansion + right_expansion
                # Branch/Gate not taken
                frontier.append((
                    child_half_wires,
                    node[1] | absorbed_gates,
                    node[2],
                    node[3],
                ))

                # Branch/Gate taken
                op_points = {CircuitPoint(cycle_index, q) for q in op.location}
                frontier.append((
                    list(set(child_half_wires + expansion)),
                    node[1] | absorbed_gates | {(cycle_index, op)},
                    node[2].union(op.location),
                    node[3] | op_points,
                ))

                # Append terminal node to handle absorbed gates with no branches
            if len(node[1] | absorbed_gates) != len(node[1]):
                frontier.append(([], node[1] | absorbed_gates, *node[2:]))

        return best_region
