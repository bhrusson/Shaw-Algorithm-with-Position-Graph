from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CIRCUITS = [
    'QFT_wsq_16_compiled',
    'QFT_wsq_32_compiled',
    'QFT_wsq_64_compiled',
    'QFT_wsq_128_compiled',
]

SUMMARY_LINE_RE = re.compile(
    r'^\s{2}(?P<label>.+?)\s{2,}'
    r'compile_time_s=(?P<compile_time_s>\S+)\s+'
    r'runtime_us=(?P<runtime_us>\S+)\s+'
    r'fidelity=(?P<fidelity>\S+)\s+'
    r'instructions=(?P<instructions>\d+)\s+'
    r'execute_rounds=(?P<execute_rounds>\S+)\s+'
    r'move_rounds=(?P<move_rounds>\S+)\s*$',
)


@dataclass
class CaseConfig:
    benchmark: str
    grid_cols: int
    grid_rows: int
    num_qudits: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run CG vs PGS compare sweeps for QFT benchmark circuits.',
    )
    parser.add_argument(
        '--circuits',
        nargs='+',
        default=DEFAULT_CIRCUITS,
        help='Benchmark circuit stems without the .qasm extension.',
    )
    parser.add_argument('--trap-capacity', type=int, default=3)
    parser.add_argument('--num-layout-passes', type=int, default=2)
    parser.add_argument('--gate-type', default='FM')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--routing-mode',
        choices=['heuristic', 'bruteforce'],
        default='bruteforce',
    )
    parser.add_argument(
        '--stage',
        choices=['layout', 'full', 'full-matched-layout'],
        default='full',
    )
    parser.add_argument('--cg-congestion-rate-override', type=float, default=None)
    parser.add_argument('--pgs-congestion-rate-override', type=float, default=None)
    parser.add_argument(
        '--pgs-move-path-modes',
        nargs='+',
        choices=['hops', 'weighted'],
        default=['hops'],
    )
    parser.add_argument(
        '--grid-strategy',
        choices=['min-square', 'fixed'],
        default='min-square',
        help='Use a per-circuit minimum square grid or one fixed grid for every run.',
    )
    parser.add_argument('--grid-cols', type=int, default=None)
    parser.add_argument('--grid-rows', type=int, default=None)
    parser.add_argument(
        '--output-root',
        default='bqskit/shuttling/qccd/batch_results',
        help='Parent directory for this sweep output.',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional output folder name. Defaults to a timestamped name.',
    )
    parser.add_argument('--timeout-seconds', type=int, default=0)
    parser.add_argument('--cooldown-seconds', type=float, default=0.0)
    parser.add_argument('--verbose-status', action='store_true')
    parser.add_argument('--print-paths', action='store_true')
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def benchmark_dir() -> Path:
    return Path('bqskit/shuttling/qccd/benchmark_circuits')


def benchmark_path(stem: str) -> Path:
    return benchmark_dir() / f'{stem}.qasm'


def extract_qudit_count(stem: str) -> int:
    match = re.search(r'_(\d+)_compiled$', stem)
    if match is None:
        raise ValueError(f'Could not infer qudit count from benchmark name: {stem}')
    return int(match.group(1))


def min_square_grid(num_qudits: int, trap_capacity: int) -> tuple[int, int]:
    required_traps = math.ceil(num_qudits / trap_capacity)
    side_traps = math.ceil(math.sqrt(required_traps))
    side = max(side_traps - 1, 1)
    return side, side


def build_case_config(stem: str, args: argparse.Namespace) -> CaseConfig:
    num_qudits = extract_qudit_count(stem)
    if args.grid_strategy == 'fixed':
        if args.grid_cols is None or args.grid_rows is None:
            raise ValueError(
                '--grid-cols and --grid-rows are required when --grid-strategy fixed.',
            )
        grid_cols = int(args.grid_cols)
        grid_rows = int(args.grid_rows)
    else:
        grid_cols, grid_rows = min_square_grid(num_qudits, args.trap_capacity)
    return CaseConfig(
        benchmark=stem,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_qudits=num_qudits,
    )


def make_output_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'qft_cg_vs_pgs_{timestamp}'
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_command(case: CaseConfig, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        '-m',
        'bqskit.shuttling.qccd.compare_shaw_grid_cg_pgs',
        case.benchmark,
        '--trap-capacity',
        str(args.trap_capacity),
        '--num-layout-passes',
        str(args.num_layout_passes),
        '--gate-type',
        args.gate_type,
        '--seed',
        str(args.seed),
        '--grid-cols',
        str(case.grid_cols),
        '--grid-rows',
        str(case.grid_rows),
        '--routing-mode',
        args.routing_mode,
        '--stage',
        args.stage,
        '--window',
        str(args.window),
        '--pgs-move-path-modes',
        *args.pgs_move_path_modes,
    ]
    if args.cg_congestion_rate_override is not None:
        cmd.extend([
            '--cg-congestion-rate-override',
            str(args.cg_congestion_rate_override),
        ])
    if args.pgs_congestion_rate_override is not None:
        cmd.extend([
            '--pgs-congestion-rate-override',
            str(args.pgs_congestion_rate_override),
        ])
    if args.verbose_status:
        cmd.append('--verbose-status')
    if args.print_paths:
        cmd.append('--print-paths')
    return cmd


def safe_float(text: str) -> float | None:
    if text == 'n/a':
        return None
    return float(text)


def safe_int(text: str) -> int | None:
    if text == 'n/a':
        return None
    return int(text)


def parse_summary(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = SUMMARY_LINE_RE.match(line)
        if match is None:
            continue
        rows.append({
            'label': match.group('label').strip(),
            'compile_time_s': safe_float(match.group('compile_time_s')),
            'runtime_us': safe_float(match.group('runtime_us')),
            'fidelity': safe_float(match.group('fidelity')),
            'instructions': int(match.group('instructions')),
            'execute_rounds': safe_int(match.group('execute_rounds')),
            'move_rounds': safe_int(match.group('move_rounds')),
        })
    return rows


def write_case_output(
    output_dir: Path,
    case: CaseConfig,
    command: list[str],
    duration_s: float,
    completed: subprocess.CompletedProcess[str] | None,
    timed_out: bool,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_slug = f'{case.benchmark}__grid_{case.grid_cols}x{case.grid_rows}'
    txt_path = output_dir / f'{case_slug}.txt'
    json_path = output_dir / f'{case_slug}.json'

    stdout = '' if completed is None else completed.stdout
    stderr = '' if completed is None else completed.stderr
    return_code = None if completed is None else completed.returncode

    txt_lines = [
        f'command: {" ".join(command)}',
        f'duration_s: {duration_s:.6f}',
        f'timed_out: {timed_out}',
        f'return_code: {return_code}',
        '',
        '--- stdout ---',
        stdout.rstrip(),
        '',
        '--- stderr ---',
        stderr.rstrip(),
        '',
    ]
    txt_path.write_text('\n'.join(txt_lines), encoding='utf-8')

    case_record = {
        'benchmark': case.benchmark,
        'num_qudits': case.num_qudits,
        'grid_cols': case.grid_cols,
        'grid_rows': case.grid_rows,
        'command': command,
        'duration_s': duration_s,
        'timed_out': timed_out,
        'return_code': return_code,
        'summary_rows': summary_rows,
        'stdout_file': txt_path.name,
    }
    json_path.write_text(json.dumps(case_record, indent=2), encoding='utf-8')
    return case_record


def write_summary_csv(output_dir: Path, case_records: list[dict[str, Any]]) -> None:
    csv_path = output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'benchmark',
                'num_qudits',
                'grid_cols',
                'grid_rows',
                'duration_s',
                'timed_out',
                'return_code',
                'label',
                'compile_time_s',
                'runtime_us',
                'fidelity',
                'instructions',
                'execute_rounds',
                'move_rounds',
                'stdout_file',
            ],
        )
        writer.writeheader()
        for case_record in case_records:
            for row in case_record['summary_rows']:
                writer.writerow({
                    'benchmark': case_record['benchmark'],
                    'num_qudits': case_record['num_qudits'],
                    'grid_cols': case_record['grid_cols'],
                    'grid_rows': case_record['grid_rows'],
                    'duration_s': case_record['duration_s'],
                    'timed_out': case_record['timed_out'],
                    'return_code': case_record['return_code'],
                    'label': row['label'],
                    'compile_time_s': row['compile_time_s'],
                    'runtime_us': row['runtime_us'],
                    'fidelity': row['fidelity'],
                    'instructions': row['instructions'],
                    'execute_rounds': row['execute_rounds'],
                    'move_rounds': row['move_rounds'],
                    'stdout_file': case_record['stdout_file'],
                })


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args)

    existing_cases: list[CaseConfig] = []
    missing_cases: list[str] = []
    for stem in args.circuits:
        if benchmark_path(stem).exists():
            existing_cases.append(build_case_config(stem, args))
        else:
            missing_cases.append(stem)

    manifest = {
        'created_at': datetime.now().isoformat(),
        'python': sys.executable,
        'args': vars(args),
        'missing_benchmarks': missing_cases,
        'cases': [asdict(case) for case in existing_cases],
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )

    if missing_cases:
        print(f'Skipping missing benchmarks: {missing_cases}')
    if not existing_cases:
        raise SystemExit('No matching benchmark circuits were found.')

    case_records: list[dict[str, Any]] = []
    timeout_seconds = None if args.timeout_seconds <= 0 else args.timeout_seconds

    for case in existing_cases:
        command = build_command(case, args)
        print(
            f'Running {case.benchmark} on grid {case.grid_cols}x{case.grid_rows} '
            f'({case.num_qudits} qudits)...'
        )

        if args.dry_run:
            case_record = write_case_output(
                output_dir,
                case,
                command,
                duration_s=0.0,
                completed=subprocess.CompletedProcess(command, 0, '', ''),
                timed_out=False,
                summary_rows=[],
            )
            case_records.append(case_record)
            continue

        start = time.perf_counter()
        timed_out = False
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                returncode=-1,
                stdout=exc.stdout or '',
                stderr=exc.stderr or '',
            )
        duration_s = time.perf_counter() - start

        summary_rows = parse_summary(completed.stdout if completed is not None else '')
        case_record = write_case_output(
            output_dir,
            case,
            command,
            duration_s,
            completed,
            timed_out,
            summary_rows,
        )
        case_records.append(case_record)

        if (timed_out or (completed is not None and completed.returncode != 0)) and not args.continue_on_error:
            write_summary_csv(output_dir, case_records)
            raise SystemExit(
                f'Run failed for {case.benchmark}. See {case_record["stdout_file"]}.',
            )

        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    write_summary_csv(output_dir, case_records)
    print(f'Wrote sweep results to {output_dir}')


if __name__ == '__main__':
    main()
