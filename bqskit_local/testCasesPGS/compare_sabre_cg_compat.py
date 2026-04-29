from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bqskit_local.testCasesPGS.ibmEagleCommon import IBM_EAGLE_NUM_QUDITS


DEFAULT_LAYOUT_PASSES = [2, 6]
DEFAULT_TIMEOUT_SECONDS = 0
BENCHMARK_DIR = Path('bqskit_local/benchmark_circuits')
RUN_METRIC_PATTERNS = {
    'architecture': re.compile(r'^Architecture: (?P<value>.+)$'),
    'num_qudits': re.compile(r'^Number of qudits: (?P<value>\d+)$'),
    'original_ops': re.compile(r'^Original operation count: (?P<value>\d+)$'),
    'compiled_ops': re.compile(r'^Compiled operation count: (?P<value>\d+)$'),
    'runtime_s': re.compile(r'^Compilation runtime \(s\): (?P<value>\d+(?:\.\d+)?)$'),
    'initial_mapping': re.compile(r'^Initial mapping: (?P<value>.+)$'),
    'final_mapping': re.compile(r'^Final mapping: (?P<value>.+)$'),
    'placement': re.compile(r'^Placement: (?P<value>.+)$'),
    'heuristic_stats': re.compile(r'^Heuristic stats JSON: (?P<value>\{.+\})$'),
}

HEURISTIC_TOTAL_FIELDS = [
    'best_swap_calls',
    'score_swap_calls',
    'frontier_size_avg_per_best_swap',
    'frontier_size_max',
    'extended_size_avg_per_best_swap',
    'extended_size_max',
    'candidate_count_avg_per_best_swap',
    'candidate_count_max',
    'affected_total_avg_per_candidate',
    'affected_total_max',
    'full_rescore_terms_avg_per_candidate',
    'full_rescore_terms_max',
    'rescore_terms_ratio',
    'affected_fraction_of_full_rescore',
]


@dataclass(frozen=True)
class CaseConfig:
    circuit: str
    num_qudits: int
    architecture: str
    layout_passes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare SABRE --cg-compat against CG over mirrored wsq benchmarks.',
    )
    parser.add_argument(
        '--circuits',
        nargs='+',
        default=None,
        help='Optional benchmark stems without .qasm. Defaults to mirrored QAOA/QFT wsq circuits up to 128 qudits.',
    )
    parser.add_argument(
        '--architectures',
        nargs='+',
        default=['min-square', 'ibm-eagle'],
        help='Architectures to compare. Use min-square for per-circuit grid-NxN and/or ibm-eagle.',
    )
    parser.add_argument(
        '--layout-passes',
        nargs='+',
        type=int,
        default=DEFAULT_LAYOUT_PASSES,
        help='Layout pass counts to compare.',
    )
    parser.add_argument(
        '--output-root',
        default='bqskit_local/results',
        help='Parent directory for comparison output.',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional output folder name. Defaults to a timestamped directory.',
    )
    parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help='Optional per-run timeout. Zero disables the timeout.',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue even if an individual run fails.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Write the planned cases without executing them.',
    )
    parser.add_argument(
        '--track-heuristic-stats',
        action='store_true',
        help='Ask child SABRE runs to emit frontier/extended-set heuristic statistics.',
    )
    parser.add_argument(
        '--profile-dir',
        default=None,
        help=(
            'Optional directory for per-child cProfile dumps. Relative paths '
            'are created under the comparison output directory.'
        ),
    )
    parser.add_argument(
        '--cg-can-exe-mode',
        choices=['optimized', 'legacy'],
        default='optimized',
        help=(
            'CG executability implementation to use. optimized uses direct '
            'adjacency/BFS; legacy uses get_subgraph(...).is_fully_connected().'
        ),
    )
    return parser.parse_args()


def extract_qudit_count(stem: str) -> int:
    match = re.search(r'_(\d+)_compiled$', stem)
    if match is None:
        raise ValueError(f'Could not infer qudit count from benchmark name: {stem}')
    return int(match.group(1))


def default_circuits() -> list[str]:
    stems: list[str] = []
    for path in sorted(BENCHMARK_DIR.glob('*_wsq_*_compiled.qasm')):
        stem = path.stem
        if not (stem.startswith('QAOA_wsq_') or stem.startswith('QFT_wsq_')):
            continue
        if extract_qudit_count(stem) <= 128:
            stems.append(stem)
    return stems


def min_square_architecture(num_qudits: int) -> str:
    side = max(1, math.ceil(math.sqrt(num_qudits)))
    return f'grid-{side}x{side}'


def expand_architectures(num_qudits: int, requested: list[str]) -> list[str]:
    expanded: list[str] = []
    for architecture in requested:
        if architecture == 'min-square':
            expanded.append(min_square_architecture(num_qudits))
            continue

        if architecture == 'ibm-eagle' and num_qudits > IBM_EAGLE_NUM_QUDITS:
            continue

        expanded.append(architecture)

    deduped: list[str] = []
    seen: set[str] = set()
    for architecture in expanded:
        if architecture not in seen:
            seen.add(architecture)
            deduped.append(architecture)
    return deduped


def build_cases(args: argparse.Namespace) -> tuple[list[CaseConfig], list[dict[str, object]]]:
    circuits = args.circuits if args.circuits is not None else default_circuits()
    cases: list[CaseConfig] = []
    skipped: list[dict[str, object]] = []

    for circuit in circuits:
        num_qudits = extract_qudit_count(circuit)
        architectures = expand_architectures(num_qudits, list(args.architectures))
        skipped_architectures = sorted(set(args.architectures) - set(architectures))
        for architecture in skipped_architectures:
            if architecture == 'ibm-eagle' and num_qudits > IBM_EAGLE_NUM_QUDITS:
                skipped.append({
                    'circuit': circuit,
                    'num_qudits': num_qudits,
                    'architecture': architecture,
                    'reason': f'requires {num_qudits} qudits but ibm-eagle has {IBM_EAGLE_NUM_QUDITS}',
                })

        for architecture in architectures:
            for layout_passes in args.layout_passes:
                cases.append(
                    CaseConfig(
                        circuit=circuit,
                        num_qudits=num_qudits,
                        architecture=architecture,
                        layout_passes=int(layout_passes),
                    ),
                )

    return cases, skipped


def make_output_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'sabre_cg_compat_compare_{timestamp}'
    else:
        run_name = args.run_name

    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    return output_dir


def build_command(
    case: CaseConfig,
    label: str,
    *,
    track_heuristic_stats: bool = False,
    profile_output: Path | None = None,
    cg_legacy_can_exe: bool = False,
) -> list[str]:
    if label == 'cg':
        command = [
            sys.executable,
            '-m',
            'bqskit_local.testCasesPGS.benchmark_sabre_cg',
            case.circuit,
            '--architecture',
            case.architecture,
            '--sabre-layout-passes',
            str(case.layout_passes),
        ]
        if track_heuristic_stats:
            command.append('--track-heuristic-stats')
        if profile_output is not None:
            command.extend(['--profile-output', str(profile_output)])
        if cg_legacy_can_exe:
            command.append('--legacy-can-exe')
        return command

    if label == 'pgs_cg_compat':
        command = [
            sys.executable,
            '-m',
            'bqskit_local.testCasesPGS.benchmark_sabre_pgs',
            case.circuit,
            '--architecture',
            case.architecture,
            '--algorithm',
            'sabre',
            '--cg-compat',
            '--sabre-layout-passes',
            str(case.layout_passes),
        ]
        if track_heuristic_stats:
            command.append('--track-heuristic-stats')
        if profile_output is not None:
            command.extend(['--profile-output', str(profile_output)])
        return command

    raise ValueError(f'Unknown run label: {label}')


def parse_metrics(stdout: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        for key, pattern in RUN_METRIC_PATTERNS.items():
            match = pattern.match(stripped)
            if match is None:
                continue
            value = match.group('value')
            if key in ('num_qudits', 'original_ops', 'compiled_ops'):
                metrics[key] = int(value)
            elif key == 'runtime_s':
                metrics[key] = float(value)
            elif key == 'heuristic_stats':
                metrics[key] = json.loads(value)
            else:
                metrics[key] = value
    return metrics


def run_command(
    command: list[str],
    *,
    timeout_seconds: int,
) -> tuple[subprocess.CompletedProcess[str] | None, bool, float]:
    start = time.perf_counter()
    timed_out = False
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=None if timeout_seconds <= 0 else timeout_seconds,
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
    return completed, timed_out, duration_s


def case_slug(case: CaseConfig) -> str:
    return f'{case.circuit}__{case.architecture}__passes_{case.layout_passes}'


def make_profile_dir(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.profile_dir is None:
        return None

    profile_dir = Path(args.profile_dir)
    if not profile_dir.is_absolute():
        profile_dir = output_dir / profile_dir
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def write_raw_output(
    output_dir: Path,
    case: CaseConfig,
    label: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str] | None,
    timed_out: bool,
    duration_s: float,
) -> str:
    raw_path = output_dir / 'raw' / f'{case_slug(case)}__{label}.txt'
    stdout = '' if completed is None else completed.stdout
    stderr = '' if completed is None else completed.stderr
    return_code = None if completed is None else completed.returncode
    raw_path.write_text(
        '\n'.join([
            f'command: {" ".join(command)}',
            f'timed_out: {timed_out}',
            f'duration_s: {duration_s:.6f}',
            f'return_code: {return_code}',
            '',
            '--- stdout ---',
            stdout.rstrip(),
            '',
            '--- stderr ---',
            stderr.rstrip(),
            '',
        ]),
        encoding='utf-8',
    )
    return raw_path.name


def compare_mappings(left: dict[str, object], right: dict[str, object], key: str) -> bool | None:
    if key not in left or key not in right:
        return None
    return left[key] == right[key]


def heuristic_total(metrics: dict[str, object]) -> dict[str, object] | None:
    heuristic_stats = metrics.get('heuristic_stats')
    if not isinstance(heuristic_stats, dict):
        return None
    total = heuristic_stats.get('total')
    return total if isinstance(total, dict) else None


def comparison_row(
    case: CaseConfig,
    cg_run: dict[str, object],
    pgs_run: dict[str, object],
) -> dict[str, object]:
    cg_metrics = cg_run['metrics']
    pgs_metrics = pgs_run['metrics']
    cg_compiled_ops = cg_metrics.get('compiled_ops')
    pgs_compiled_ops = pgs_metrics.get('compiled_ops')
    cg_runtime_s = cg_metrics.get('runtime_s')
    pgs_runtime_s = pgs_metrics.get('runtime_s')

    delta_ops = None
    ops_ratio = None
    if isinstance(cg_compiled_ops, int) and isinstance(pgs_compiled_ops, int):
        delta_ops = pgs_compiled_ops - cg_compiled_ops
        if cg_compiled_ops != 0:
            ops_ratio = pgs_compiled_ops / cg_compiled_ops

    delta_runtime_s = None
    runtime_ratio = None
    if isinstance(cg_runtime_s, float) and isinstance(pgs_runtime_s, float):
        delta_runtime_s = pgs_runtime_s - cg_runtime_s
        if cg_runtime_s != 0:
            runtime_ratio = pgs_runtime_s / cg_runtime_s

    cg_total_heuristic = heuristic_total(cg_metrics)
    pgs_total_heuristic = heuristic_total(pgs_metrics)

    row = {
        'circuit': case.circuit,
        'num_qudits': case.num_qudits,
        'architecture': case.architecture,
        'layout_passes': case.layout_passes,
        'cg_status': cg_run['status'],
        'pgs_status': pgs_run['status'],
        'cg_runtime_s': cg_runtime_s,
        'pgs_runtime_s': pgs_runtime_s,
        'delta_runtime_s': delta_runtime_s,
        'runtime_ratio': runtime_ratio,
        'cg_compiled_ops': cg_compiled_ops,
        'pgs_compiled_ops': pgs_compiled_ops,
        'delta_ops': delta_ops,
        'ops_ratio': ops_ratio,
        'same_compiled_ops': (
            cg_compiled_ops == pgs_compiled_ops
            if isinstance(cg_compiled_ops, int) and isinstance(pgs_compiled_ops, int)
            else None
        ),
        'same_final_mapping': compare_mappings(cg_metrics, pgs_metrics, 'final_mapping'),
        'same_placement': compare_mappings(cg_metrics, pgs_metrics, 'placement'),
        'cg_raw_output': cg_run['raw_output'],
        'pgs_raw_output': pgs_run['raw_output'],
    }

    for prefix, stats in (('cg', cg_total_heuristic), ('pgs', pgs_total_heuristic)):
        for field in HEURISTIC_TOTAL_FIELDS:
            row[f'{prefix}_{field}'] = None if stats is None else stats.get(field)

    row['same_total_heuristic_stats'] = (
        cg_total_heuristic == pgs_total_heuristic
        if cg_total_heuristic is not None and pgs_total_heuristic is not None
        else None
    )
    return row


def summary_fieldnames() -> list[str]:
    fieldnames = [
        'circuit',
        'num_qudits',
        'architecture',
        'layout_passes',
        'cg_status',
        'pgs_status',
        'cg_runtime_s',
        'pgs_runtime_s',
        'delta_runtime_s',
        'runtime_ratio',
        'cg_compiled_ops',
        'pgs_compiled_ops',
        'delta_ops',
        'ops_ratio',
        'same_compiled_ops',
        'same_final_mapping',
        'same_placement',
        'same_total_heuristic_stats',
    ]
    for prefix in ('cg', 'pgs'):
        for field in HEURISTIC_TOTAL_FIELDS:
            fieldnames.append(f'{prefix}_{field}')
    fieldnames.extend([
        'cg_raw_output',
        'pgs_raw_output',
    ])
    return fieldnames


def write_summary_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    csv_path = output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fieldnames(),
        )
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    if value is None:
        return ''
    return str(value)


def write_markdown_table(
    output_dir: Path,
    rows: list[dict[str, object]],
    skipped: list[dict[str, object]],
) -> None:
    md_path = output_dir / 'summary.md'
    lines = [
        '# SABRE --cg-compat vs CG',
        '',
        '| Circuit | Qudits | Architecture | Passes | CG ops | PGS ops | Delta ops | Same ops | Same stats | CG time (s) | PGS time (s) | Time ratio |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |',
    ]

    for row in rows:
        lines.append(
            '| '
            f"{row['circuit']} | "
            f"{row['num_qudits']} | "
            f"{row['architecture']} | "
            f"{row['layout_passes']} | "
            f"{row['cg_compiled_ops'] if row['cg_compiled_ops'] is not None else ''} | "
            f"{row['pgs_compiled_ops'] if row['pgs_compiled_ops'] is not None else ''} | "
            f"{row['delta_ops'] if row['delta_ops'] is not None else ''} | "
            f"{row['same_compiled_ops'] if row['same_compiled_ops'] is not None else ''} | "
            f"{row['same_total_heuristic_stats'] if row['same_total_heuristic_stats'] is not None else ''} | "
            f"{format_float(row['cg_runtime_s'])} | "
            f"{format_float(row['pgs_runtime_s'])} | "
            f"{format_float(row['runtime_ratio'])} |"
        )

    if skipped:
        lines.extend([
            '',
            '## Skipped Cases',
            '',
            '| Circuit | Qudits | Architecture | Reason |',
            '| --- | ---: | --- | --- |',
        ])
        for item in skipped:
            lines.append(
                f"| {item['circuit']} | {item['num_qudits']} | {item['architecture']} | {item['reason']} |"
            )

    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    cases, skipped = build_cases(args)
    output_dir = make_output_dir(args)
    profile_dir = make_profile_dir(args, output_dir)

    manifest = {
        'created_at': datetime.now().isoformat(),
        'python': sys.executable,
        'args': vars(args),
        'cases': [
            {
                'circuit': case.circuit,
                'num_qudits': case.num_qudits,
                'architecture': case.architecture,
                'layout_passes': case.layout_passes,
            }
            for case in cases
        ],
        'skipped': skipped,
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )

    rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for case_index, case in enumerate(cases):
        print(
            f'Running {case.circuit} on {case.architecture} with '
            f'{case.layout_passes} layout passes...'
        )
        case_runs: dict[str, dict[str, object]] = {}
        for label in ('cg', 'pgs_cg_compat'):
            profile_output = (
                profile_dir / f'{case_index:04d}__{case_slug(case)}__{label}.prof'
                if profile_dir is not None
                else None
            )
            command = build_command(
                case,
                label,
                track_heuristic_stats=args.track_heuristic_stats,
                profile_output=profile_output,
                cg_legacy_can_exe=args.cg_can_exe_mode == 'legacy',
            )
            if args.dry_run:
                completed = subprocess.CompletedProcess(command, 0, '', '')
                timed_out = False
                duration_s = 0.0
            else:
                completed, timed_out, duration_s = run_command(
                    command,
                    timeout_seconds=args.timeout_seconds,
                )

            raw_output = write_raw_output(
                output_dir,
                case,
                label,
                command,
                completed,
                timed_out,
                duration_s,
            )
            metrics = parse_metrics('' if completed is None else completed.stdout)
            status = 'ok'
            if timed_out:
                status = 'timed_out'
            elif completed is None or completed.returncode != 0:
                status = 'failed'

            run_record = {
                'circuit': case.circuit,
                'num_qudits': case.num_qudits,
                'architecture': case.architecture,
                'layout_passes': case.layout_passes,
                'label': label,
                'status': status,
                'duration_s': duration_s,
                'return_code': None if completed is None else completed.returncode,
                'timed_out': timed_out,
                'command': command,
                'profile_output': None if profile_output is None else str(profile_output),
                'metrics': metrics,
                'raw_output': raw_output,
            }
            run_records.append(run_record)
            case_runs[label] = run_record

            if status != 'ok' and not args.continue_on_error:
                (output_dir / 'runs.json').write_text(
                    json.dumps(run_records, indent=2),
                    encoding='utf-8',
                )
                raise SystemExit(
                    f'Run failed for {case.circuit} on {case.architecture} '
                    f'({label}, passes={case.layout_passes}).',
                )

        rows.append(comparison_row(case, case_runs['cg'], case_runs['pgs_cg_compat']))

    (output_dir / 'runs.json').write_text(
        json.dumps(run_records, indent=2),
        encoding='utf-8',
    )
    write_summary_csv(output_dir, rows)
    write_markdown_table(output_dir, rows, skipped)
    print(f'Wrote comparison results to {output_dir}')


if __name__ == '__main__':
    main()
