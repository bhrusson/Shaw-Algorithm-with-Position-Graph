from __future__ import annotations

from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.shuttling.QCCD_schedule_new import print_event_trace
from bqskit.shuttling.QCCD_schedule_new import schedule_qccd_from_instructions_v3


def _normalize_assignment(
    assignment: dict[int, int],
) -> dict[int, int]:
    return {
        int(logical): int(position)
        for logical, position in assignment.items()
    }


def schedule_QCCD_CG(
    instructions_list: list,
    circuit: Circuit,
    initial_mapping: list[int] | None,
    initial_ion_assignment: dict[int, int],
    qccd_machine: Any,
    *,
    full_initial_ion_assignment: dict[int, int] | None = None,
    parallel: bool = True,
    background_heating_rate: float = 1.0,
    base_gate_fidelity: float = 0.992,
    min_gate_fidelity: float = 1e-4,
    validate_instruction_cost: bool = True,
    honeywell_mode: bool = True,
    intra_trap_swap_mode: str = 'gate',
) -> dict[str, Any]:
    """
    Schedule a CG instruction stream using the Brent notebook scheduler model.

    `Brent_scheduling.ipynb` introduced the `schedule_qccd_from_instructions_v3`
    scheduler. This module provides a CG-oriented entrypoint that preserves the
    legacy `schedule_QCCD(...)` call shape while delegating to the maintained
    implementation in `bqskit.shuttling.QCCD_schedule_new`.
    """
    del initial_mapping

    normalized_initial = _normalize_assignment(initial_ion_assignment)
    normalized_full = (
        None
        if full_initial_ion_assignment is None
        else _normalize_assignment(full_initial_ion_assignment)
    )

    return schedule_qccd_from_instructions_v3(
        instruction_lst=instructions_list,
        initial_ion_assignment=normalized_initial,
        full_initial_ion_assignment=normalized_full,
        machine_model=qccd_machine,
        circuit=circuit,
        parallel=parallel,
        background_heating_rate=background_heating_rate,
        base_gate_fidelity=base_gate_fidelity,
        min_gate_fidelity=min_gate_fidelity,
        validate_instruction_cost=validate_instruction_cost,
        honeywell_mode=honeywell_mode,
        intra_trap_swap_mode=intra_trap_swap_mode,
        execute_location_mode='logical',
    )


__all__ = [
    'print_event_trace',
    'schedule_QCCD_CG',
]
