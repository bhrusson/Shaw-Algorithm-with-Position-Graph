from __future__ import annotations
import cProfile
import pstats
from collections.abc import Mapping
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from bqskit.compiler.passdata import PassData

from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.position_graph_state_PGS import PositionGraphState

FULL_ASSIGNMENT_KEY = 'full_ion_assignment_qccd_pgs'
PROGRAM_ASSIGNMENT_KEY = 'program_ion_assignment_qccd'
PROGRAM_ION_IDS_KEY = 'program_ion_ids_qccd'
T = TypeVar('T')


def profiled_call(
    profile_dir: Path | None,
    stem: str,
    sort_key: str,
    func: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    if profile_dir is None:
        return func(*args, **kwargs)

    profile_dir.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    try:
        return profiler.runcall(func, *args, **kwargs)
    finally:
        prof_path = profile_dir / f'{stem}.prof'
        txt_path = profile_dir / f'{stem}.prof.txt'
        profiler.dump_stats(str(prof_path))
        with txt_path.open('w', encoding='utf-8') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats(sort_key).print_stats()


def _normalize_assignment(
    ion_assignment: Mapping[int, int] | None,
) -> dict[int, int]:
    if ion_assignment is None:
        return {}

    return {int(logical): int(position) for logical, position in ion_assignment.items()}


def _normalize_program_ion_ids(
    program_ion_ids: object,
    fallback_assignment: Mapping[int, int] | None,
) -> list[int]:
    if program_ion_ids is None:
        if fallback_assignment is None:
            return []
        return sorted(int(logical) for logical in fallback_assignment.keys())

    return sorted(int(logical) for logical in program_ion_ids)


def full_assignment_from_pgs(
    pgs: PositionGraphState,
) -> dict[int, int]:
    return {
        int(logical): int(position)
        for logical, position in enumerate(pgs.logical_to_position)
        if int(position) != -1
    }


def program_assignment_from_pgs(
    pgs: PositionGraphState,
    program_ion_ids: list[int],
) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for logical in program_ion_ids:
        position = int(pgs.logical_to_position[int(logical)])
        if position == -1:
            raise RuntimeError(f'Program ion {logical} is not placed in PositionGraphState.')
        assignment[int(logical)] = position
    return assignment


def export_pgs_views_to_passdata(
    data: PassData,
    pgs: PositionGraphState,
    *,
    assignment_key: str = 'ion_assignment_qccd',
    full_assignment_key: str = FULL_ASSIGNMENT_KEY,
    program_assignment_key: str = PROGRAM_ASSIGNMENT_KEY,
    program_ion_ids_key: str = PROGRAM_ION_IDS_KEY,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    program_ion_ids = _normalize_program_ion_ids(
        data.get(program_ion_ids_key),
        data.get(program_assignment_key, data.get(assignment_key)),
    )
    full_assignment = full_assignment_from_pgs(pgs)
    program_assignment = program_assignment_from_pgs(pgs, program_ion_ids)

    data[program_ion_ids_key] = list(program_ion_ids)
    data[full_assignment_key] = dict(full_assignment)
    data[program_assignment_key] = dict(program_assignment)
    data[assignment_key] = dict(program_assignment)
    data[f'{assignment_key}_pgs'] = pgs

    return full_assignment, program_assignment, program_ion_ids


def build_pgs_from_passdata(
    machine_model: QCCDMachineModel,
    data: PassData,
    *,
    assignment_key: str = 'ion_assignment_qccd',
    full_assignment_key: str = FULL_ASSIGNMENT_KEY,
    program_assignment_key: str = PROGRAM_ASSIGNMENT_KEY,
    program_ion_ids_key: str = PROGRAM_ION_IDS_KEY,
) -> PositionGraphState:
    """
    Build a PositionGraphState for QCCD passes from standard PassData keys.

    This mirrors the role that placement/build helpers play in the SABRE PGS
    passes. The initial expectation is that a workflow stores the logical ion
    placement in ``data[assignment_key]`` before the PGS passes run.
    """

    if not isinstance(machine_model, QCCDMachineModel):
        raise TypeError(
            f'Expected QCCDMachineModel, got {type(machine_model)}.',
        )

    program_assignment = _normalize_assignment(
        data.get(program_assignment_key, data.get(assignment_key)),
    )
    program_ion_ids = _normalize_program_ion_ids(
        data.get(program_ion_ids_key),
        program_assignment,
    )
    full_assignment = _normalize_assignment(data.get(full_assignment_key))
    pgs_cache_key = f'{assignment_key}_pgs'
    cached_pgs = data.get(pgs_cache_key)

    if (
        isinstance(cached_pgs, PositionGraphState)
        and cached_pgs.position_graph is machine_model.position_graph
        and tuple(cached_pgs.radices) == tuple(machine_model.radixes)
    ):
        pgs = cached_pgs
    else:
        pgs = machine_model.build_pgs_from_assignments(
            program_assignment,
            full_assignment if full_assignment else None,
        )

    data[program_ion_ids_key] = list(program_ion_ids)
    data[program_assignment_key] = program_assignment_from_pgs(pgs, program_ion_ids)
    data[full_assignment_key] = full_assignment_from_pgs(pgs)
    data[pgs_cache_key] = pgs
    return pgs
