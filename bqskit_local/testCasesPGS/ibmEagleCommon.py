from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit_local.position.graph import (
    EdgeCapability,
    EdgeLabel,
    PositionCapability,
    PositionGraph,
    PositionLabel,
)

# Official IBM Washington fake-backend coupling map from Qiskit's fake provider.
# Washington is a 127-qubit Eagle-generation device.
IBM_EAGLE_NUM_QUDITS = 127
IBM_EAGLE_COUPLING_MAP = [
    [0, 1], [0, 14], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4],
    [4, 3], [4, 5], [4, 15], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6],
    [7, 8], [8, 7], [8, 16], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12],
    [12, 11], [12, 13], [12, 17], [13, 12], [14, 0], [14, 18], [15, 4], [15, 22],
    [16, 8], [16, 26], [17, 12], [17, 30], [18, 14], [18, 19], [19, 18], [19, 20],
    [20, 19], [20, 21], [20, 33], [21, 20], [21, 22], [22, 15], [22, 21], [22, 23],
    [23, 22], [23, 24], [24, 23], [24, 25], [24, 34], [25, 24], [25, 26], [26, 16],
    [26, 25], [26, 27], [27, 26], [27, 28], [28, 27], [28, 29], [28, 35], [29, 28],
    [29, 30], [30, 17], [30, 29], [30, 31], [31, 30], [31, 32], [32, 31], [32, 36],
    [33, 20], [33, 39], [34, 24], [34, 43], [35, 28], [35, 47], [36, 32], [36, 51],
    [37, 38], [37, 52], [38, 37], [38, 39], [39, 33], [39, 38], [39, 40], [40, 39],
    [40, 41], [41, 40], [41, 42], [41, 53], [42, 41], [42, 43], [43, 34], [43, 42],
    [43, 44], [44, 43], [44, 45], [45, 44], [45, 46], [45, 54], [46, 45], [46, 47],
    [47, 35], [47, 46], [47, 48], [48, 47], [48, 49], [49, 48], [49, 50], [49, 55],
    [50, 49], [50, 51], [51, 36], [51, 50], [52, 37], [52, 56], [53, 41], [53, 60],
    [54, 45], [54, 64], [55, 49], [55, 68], [56, 52], [56, 57], [57, 56], [57, 58],
    [58, 57], [58, 59], [58, 71], [59, 58], [59, 60], [60, 53], [60, 59], [60, 61],
    [61, 60], [61, 62], [62, 61], [62, 63], [62, 72], [63, 62], [63, 64], [64, 54],
    [64, 63], [64, 65], [65, 64], [65, 66], [66, 65], [66, 67], [66, 73], [67, 66],
    [67, 68], [68, 55], [68, 67], [68, 69], [69, 68], [69, 70], [70, 69], [70, 74],
    [71, 58], [71, 77], [72, 62], [72, 81], [73, 66], [73, 85], [74, 70], [74, 89],
    [75, 76], [75, 90], [76, 75], [76, 77], [77, 71], [77, 76], [77, 78], [78, 77],
    [78, 79], [79, 78], [79, 80], [79, 91], [80, 79], [80, 81], [81, 72], [81, 80],
    [81, 82], [82, 81], [82, 83], [83, 82], [83, 84], [83, 92], [84, 83], [84, 85],
    [85, 73], [85, 84], [85, 86], [86, 85], [86, 87], [87, 86], [87, 88], [87, 93],
    [88, 87], [88, 89], [89, 74], [89, 88], [90, 75], [90, 94], [91, 79], [91, 98],
    [92, 83], [92, 102], [93, 87], [93, 106], [94, 90], [94, 95], [95, 94], [95, 96],
    [96, 95], [96, 97], [96, 109], [97, 96], [97, 98], [98, 91], [98, 97], [98, 99],
    [99, 98], [99, 100], [100, 99], [100, 101], [100, 110], [101, 100], [101, 102],
    [102, 92], [102, 101], [102, 103], [103, 102], [103, 104], [104, 103], [104, 105],
    [104, 111], [105, 104], [105, 106], [106, 93], [106, 105], [106, 107], [107, 106],
    [107, 108], [108, 107], [108, 112], [109, 96], [110, 100], [110, 118], [111, 104],
    [111, 122], [112, 108], [112, 126], [113, 114], [114, 113], [114, 115], [115, 114],
    [115, 116], [116, 115], [116, 117], [117, 116], [117, 118], [118, 110], [118, 117],
    [118, 119], [119, 118], [119, 120], [120, 119], [120, 121], [121, 120], [121, 122],
    [122, 111], [122, 121], [122, 123], [123, 122], [123, 124], [124, 123], [124, 125],
    [125, 124], [125, 126], [126, 112], [126, 125],
]

IBM_EAGLE_UNDIRECTED_COUPLING_MAP = sorted({
    tuple(sorted((int(u), int(v))))
    for u, v in IBM_EAGLE_COUPLING_MAP
    if int(u) != int(v)
})


def build_eagle_test_circuit() -> Circuit:
    circ = Circuit(IBM_EAGLE_NUM_QUDITS)

    # A moderately deep nonlocal pattern to exercise layout and routing
    # without exploding runtime like an all-pairs benchmark.
    gate_pairs = [
        # Long-range fanout from the low end of the device.
        (0, 126),
        (0, 14),
        (0, 41),
        (0, 71),
        (0, 110),

        # Walk a path through the middle of the machine.
        (14, 18),
        (18, 33),
        (33, 39),
        (39, 53),
        (53, 71),
        (71, 90),
        (90, 110),
        (110, 126),

        # Cross-links that force alternative routing choices.
        (14, 54),
        (18, 58),
        (33, 72),
        (39, 81),
        (41, 54),
        (45, 64),
        (58, 72),
        (62, 81),
        (66, 85),
        (71, 91),
        (74, 89),
        (79, 98),
        (83, 102),
        (87, 106),
        (94, 118),
        (98, 122),
        (102, 126),

        # Reverse pressure from the high end back into the graph.
        (126, 112),
        (126, 73),
        (126, 45),
        (112, 87),
        (87, 45),
        (73, 41),

        # A few local gates to create interleaving opportunities.
        (20, 21),
        (41, 42),
        (58, 59),
        (79, 80),
        (100, 101),
        (121, 122),
    ]

    for control, target in gate_pairs:
        circ.append_gate(CNOTGate(), (control, target))

    return circ


def build_eagle_position_graph() -> PositionGraph:
    pos_label = PositionLabel(
        capability=(
            PositionCapability.EXECUTE
            | PositionCapability.MEASURE
            | PositionCapability.STARTING
        ),
        weights={
            PositionCapability.EXECUTE: 1.0,
            PositionCapability.MEASURE: 1.0,
            PositionCapability.STARTING: 1.0,
        },
    )

    edge_label = EdgeLabel(
        capability=(
            EdgeCapability.MOVE
            | EdgeCapability.SWAP
            | EdgeCapability.EXECUTE
        ),
        weights={
            EdgeCapability.MOVE: 1.0,
            EdgeCapability.SWAP: 1.0,
            EdgeCapability.EXECUTE: 1.0,
        },
    )

    pos_labels = [pos_label for _ in range(IBM_EAGLE_NUM_QUDITS)]
    edge_labels = {
        (int(u), int(v)): edge_label
        for u, v in IBM_EAGLE_COUPLING_MAP
    }

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)
