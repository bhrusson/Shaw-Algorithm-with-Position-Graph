"""Run a minimal QCCD SHAW/PGS example from the repository root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo_root / 'run_qccd.py'),
        'QAOA_wsq_8_compiled',
        '--algorithm',
        'shaw',
        '--grid-cols',
        '3',
        '--grid-rows',
        '3',
        '--trap-capacity',
        '3',
        '--num-layout-passes',
        '1',
        '--summary-only',
    ]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == '__main__':
    raise SystemExit(main())
