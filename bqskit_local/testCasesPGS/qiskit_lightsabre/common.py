from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from bqskit.ir.circuit import Circuit as BQSKITCircuit
from bqskit.ir.gates import CNOTGate as BQSKITCNOTGate
from bqskit.ir.gates import HGate as BQSKITHGate


def require_qiskit() -> tuple[Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler import CouplingMap, PassManager
        from qiskit.transpiler.passes import SabreLayout
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Qiskit is not installed in this Python environment. '
            'Install qiskit, then rerun this script.',
        ) from exc

    return QuantumCircuit, CouplingMap, PassManager, SabreLayout


def bqskit_to_qiskit_circuit(circuit: BQSKITCircuit) -> Any:
    QuantumCircuit, _, _, _ = require_qiskit()
    qc = QuantumCircuit(circuit.num_qudits)

    for op in circuit:
        if isinstance(op.gate, BQSKITHGate):
            qc.h(int(op.location[0]))
        elif isinstance(op.gate, BQSKITCNOTGate):
            qc.cx(int(op.location[0]), int(op.location[1]))
        else:
            raise NotImplementedError(
                f'Unsupported gate for Qiskit conversion: {type(op.gate)}',
            )

    return qc


@dataclass
class QiskitRunResult:
    mode: str
    runtime_s: float
    original_ops: int
    compiled_ops: int
    swap_count: int
    depth: int
    compiled_circuit: Any


def run_qiskit_sabre(
    circuit: BQSKITCircuit,
    coupling_edges: Iterable[tuple[int, int]],
    *,
    mode: str,
    seed: int | None = None,
    max_iterations: int = 3,
    layout_trials: int | None = None,
    swap_trials: int | None = None,
) -> QiskitRunResult:
    _, CouplingMap, PassManager, SabreLayout = require_qiskit()
    qiskit_circuit = bqskit_to_qiskit_circuit(circuit)
    coupling_map = CouplingMap(list(coupling_edges))

    mode_key = mode.strip().lower()
    if mode_key == 'sabre':
        # LightSABRE paper's Qiskit SABRE-compatible setting.
        selected_layout_trials = 5 if layout_trials is None else layout_trials
        selected_swap_trials = 1 if swap_trials is None else swap_trials
    elif mode_key == 'lightsabre':
        selected_layout_trials = layout_trials
        selected_swap_trials = swap_trials
    else:
        raise ValueError(f'Unsupported mode: {mode}.')

    sabre_layout = SabreLayout(
        coupling_map,
        seed=seed,
        max_iterations=max_iterations,
        layout_trials=selected_layout_trials,
        swap_trials=selected_swap_trials,
    )
    pass_manager = PassManager([sabre_layout])

    start_time = perf_counter()
    compiled = pass_manager.run(qiskit_circuit)
    elapsed_time = perf_counter() - start_time

    swap_count = sum(
        1
        for instruction in compiled.data
        if instruction.operation.name == 'swap'
    )

    return QiskitRunResult(
        mode=mode_key,
        runtime_s=elapsed_time,
        original_ops=qiskit_circuit.size(),
        compiled_ops=compiled.size(),
        swap_count=swap_count,
        depth=compiled.depth(),
        compiled_circuit=compiled,
    )


def print_result(result: QiskitRunResult) -> None:
    print(f'Mode: {result.mode}')
    print('Compilation runtime (s):', f'{result.runtime_s:.3f}')
    print('Original operation count:', result.original_ops)
    print('Compiled operation count:', result.compiled_ops)
    print('Inserted swap count:', result.swap_count)
    print('Circuit depth:', result.depth)
