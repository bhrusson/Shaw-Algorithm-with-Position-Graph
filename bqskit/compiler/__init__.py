"""
================================================
Compiler Infrastructure (:mod:`bqskit.compiler`)
================================================

Local bridge package for compiler overrides. This mirrors the installed
`bqskit.compiler` package while allowing repo-local modules to shadow
selected implementations.
"""
from __future__ import annotations

from pathlib import Path
from site import getsitepackages
from site import getusersitepackages
from typing import Any

_LOCAL_COMPILER_DIR = Path(__file__).resolve().parent

__path__ = [str(_LOCAL_COMPILER_DIR)]
_candidate_roots = [Path(getusersitepackages())]
_candidate_roots.extend(Path(path) for path in getsitepackages())
for root in _candidate_roots:
    candidate = root / 'bqskit' / 'compiler'
    if candidate.is_dir() and candidate.resolve() != _LOCAL_COMPILER_DIR:
        candidate_str = str(candidate)
        if candidate_str not in __path__:
            __path__.append(candidate_str)

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.status import CompilationStatus
from bqskit.compiler.task import CompilationTask
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike


def __getattr__(name: str) -> Any:
    if name == 'compile':
        from bqskit.compiler.compile import compile
        return compile

    raise AttributeError(f'module {__name__} has no attribute {name}')


__all__ = [
    'BasePass',
    'compile',
    'Compiler',
    'GateSet',
    'GateSetLike',
    'MachineModel',
    'PassData',
    'CompilationStatus',
    'CompilationTask',
    'Workflow',
    'WorkflowLike',
]
