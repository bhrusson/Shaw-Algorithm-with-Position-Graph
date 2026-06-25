from __future__ import annotations

import re

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate


SYNTHETIC_MULTI35_RE = re.compile(
    r'^MULTI35(?:_d(?P<layers>\d+))?_wsq_(?P<num_qudits>\d+)_compiled$',
)


def parse_synthetic_multi35_name(stem: str) -> tuple[int, int] | None:
    """Return ``(num_qudits, layers)`` for synthetic 3/5-qudit circuits."""
    match = SYNTHETIC_MULTI35_RE.match(stem)
    if match is None:
        return None

    num_qudits = int(match.group('num_qudits'))
    layers_raw = match.group('layers')
    layers = int(layers_raw) if layers_raw is not None else 8
    return num_qudits, layers


def is_synthetic_multi35_name(stem: str) -> bool:
    return parse_synthetic_multi35_name(stem) is not None


def build_synthetic_multi35_circuit(
    num_qudits: int,
    layers: int,
) -> Circuit:
    """
    Build a deterministic circuit containing only 3- and 5-qudit gates.

    Each layer partitions most of the qudits into alternating 3- and 5-qudit
    operations. The starting offset changes each layer so the front layer is
    wide, but dependencies still move across the logical register over time.
    """
    if num_qudits < 5:
        raise ValueError('Synthetic MULTI35 circuits require at least 5 qudits.')
    if layers < 1:
        raise ValueError('Synthetic MULTI35 circuits require at least one layer.')

    circuit = Circuit(num_qudits)
    gates = {
        3: VariableUnitaryGate(3),
        5: VariableUnitaryGate(5),
    }
    params = {
        arity: [0.0] * gate.num_params
        for arity, gate in gates.items()
    }

    for layer in range(layers):
        offset = (layer * 7) % num_qudits
        ordered = list(range(offset, num_qudits)) + list(range(offset))
        index = 0
        use_five = layer % 2 == 0

        while index + 3 <= num_qudits:
            arity = 5 if use_five else 3
            if index + arity > num_qudits:
                arity = 3
            if index + arity > num_qudits:
                break

            location = tuple(ordered[index:index + arity])
            circuit.append_gate(gates[arity], location, params[arity])
            index += arity
            use_five = not use_five

    return circuit


def load_synthetic_multi35_circuit(stem: str) -> Circuit | None:
    parsed = parse_synthetic_multi35_name(stem)
    if parsed is None:
        return None

    num_qudits, layers = parsed
    return build_synthetic_multi35_circuit(num_qudits, layers)
