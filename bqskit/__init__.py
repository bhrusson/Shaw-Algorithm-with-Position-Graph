"""Bridge local shuttling extensions with the installed bqskit package."""
from __future__ import annotations

from pathlib import Path
from site import getsitepackages
from site import getusersitepackages
from typing import Any

_LOCAL_BQSKIT_DIR = Path(__file__).resolve().parent

# Search local extensions first, then fall back to the installed bqskit package.
__path__ = [str(_LOCAL_BQSKIT_DIR)]
_candidate_roots = [Path(getusersitepackages())]
_candidate_roots.extend(Path(path) for path in getsitepackages())
for root in _candidate_roots:
    candidate = root / 'bqskit'
    if candidate.is_dir() and candidate.resolve() != _LOCAL_BQSKIT_DIR:
        candidate_str = str(candidate)
        if candidate_str not in __path__:
            __path__.append(candidate_str)

from bqskit._logging import disable_logging
from bqskit._logging import enable_logging
from bqskit._version import __version__  # noqa: F401
from bqskit._version import __version_info__  # noqa: F401


def __getattr__(name: str) -> Any:
    """Mirror the installed bqskit package's lazy public API."""
    if name == 'compile':
        from bqskit.compiler.compile import compile
        return compile

    if name == 'Circuit':
        from bqskit.ir.circuit import Circuit
        return Circuit

    if name == 'MachineModel':
        from bqskit.compiler.machine import MachineModel
        return MachineModel

    raise AttributeError(f'module {__name__} has no attribute {name}')


__all__ = [
    'compile',
    'MachineModel',
    'Circuit',
    'enable_logging',
    'disable_logging',
]
