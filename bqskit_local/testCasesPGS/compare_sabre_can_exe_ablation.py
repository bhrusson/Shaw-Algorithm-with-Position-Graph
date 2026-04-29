from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bqskit_local.testCasesPGS.compare_sabre_cg_compat import (
    CaseConfig,
    build_cases,
    build_command,
    case_slug,
    make_profile_dir,
    parse_metrics,
    run_command,
)


PROFILE_PATTERNS = {
    'can_exe': '_can_exe',
    'get_subgraph': 'get_subgraph',
    'is_fully_connected': 'is_fully_connected',
    'are_physical_qudits_connected': '_are_physical_qudits_connected',
}

PROFILE_LINE_RE = re.compile(
    r'^\s*(?P<ncalls>\S+)\s+'
    r'(?P<tottime>[\d.]+)\s+'
    r'(?P<percall_1>[\d.]+)\s+'
    r'(?P<cumtime>[\d.]+)\s+'
    r'(?P<percall_2>[\d.]+)\s+'
    r'(?P<func>.+)$',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run a controlled SABRE _can_exe ablation: legacy CG, '
            'optimized CG, and PGS cg-compat.'
        ),
    )
    parser.add_argument(
        '--circuits',
        nargs='+',
        required=True,
        help='Benchmark stems without .qasm.',
    )
    parser.add_argument(
        '--architectures',
        nargs='+',
        default=['min-square'],
        help='Architectures to compare. Use min-square for per-circuit grid-NxN.',
    )
    parser.add_argument(
        '--layout-passes',
        nargs='+',
        type=int,
        default=[2],
        help='Layout pass counts to compare.',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help='Number of repeated runs per case/backend.',
    )
    parser.add_argument(
        '--output-root',
        default='bqskit_local/results',
        help='Parent directory for ablation output.',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional output folder name.',
    )
    parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=0,
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
        help='Write the planned runs without executing them.',
    )
    parser.add_argument(
        '--profile-dir',
        default='profiles',
        help='Directory for cProfile dumps, relative to the output directory.',
    )
    parser.add_argument(
        '--track-heuristic-stats',
        action='store_true',
        help='Ask child SABRE runs to emit heuristic statistics as JSON.',
    )
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    run_name = (
        args.run_name
        if args.run_name is not None
        else f'sabre_can_exe_ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    return output_dir


def run_label_to_command_label(label: str) -> str:
    if label in ('cg_legacy', 'cg_optimized'):
        return 'cg'
    if label == 'pgs_cg_compat':
        return 'pgs_cg_compat'
    raise ValueError(f'Unknown ablation label: {label}')


def parse_profile_summary(profile_output: Path | None) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in PROFILE_PATTERNS:
        summary[f'{key}_calls'] = None
        summary[f'{key}_cum_s'] = None

    if profile_output is None:
        return summary

    txt_path = profile_output.with_suffix(profile_output.suffix + '.txt')
    if not txt_path.exists():
        return summary

    totals: dict[str, tuple[int, float]] = {
        key: (0, 0.0)
        for key in PROFILE_PATTERNS
    }
    for line in txt_path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = PROFILE_LINE_RE.match(line)
        if match is None:
            continue

        func = match.group('func')
        ncalls_raw = match.group('ncalls').split('/')[0]
        try:
            ncalls = int(ncalls_raw)
            cumtime = float(match.group('cumtime'))
        except ValueError:
            continue

        for key, pattern in PROFILE_PATTERNS.items():
            if pattern in func:
                old_calls, old_cumtime = totals[key]
                totals[key] = (old_calls + ncalls, old_cumtime + cumtime)

    for key, (calls, cumtime) in totals.items():
        if calls == 0:
            continue
        summary[f'{key}_calls'] = calls
        summary[f'{key}_cum_s'] = cumtime
    return summary


def write_raw_output(
    output_dir: Path,
    repeat_index: int,
    case: CaseConfig,
    label: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str] | None,
    timed_out: bool,
    duration_s: float,
) -> str:
    raw_name = f'repeat_{repeat_index + 1}__{case_slug(case)}__{label}.txt'
    raw_path = output_dir / 'raw' / raw_name
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
    return raw_name


def fieldnames() -> list[str]:
    names = [
        'repeat',
        'circuit',
        'num_qudits',
        'architecture',
        'layout_passes',
        'label',
        'status',
        'runtime_s',
        'duration_s',
        'compiled_ops',
        'initial_mapping',
        'final_mapping',
        'placement',
    ]
    for key in PROFILE_PATTERNS:
        names.append(f'{key}_calls')
        names.append(f'{key}_cum_s')
    names.extend([
        'raw_output',
        'profile_output',
        'command',
    ])
    return names


def write_summary_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(output_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        '# SABRE can_exe Ablation',
        '',
        '| Repeat | Circuit | Qudits | Arch | Passes | Label | Runtime (s) | can_exe (s) | get_subgraph (s) | Calls | Ops |',
        '| ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| '
            f"{row['repeat']} | "
            f"{row['circuit']} | "
            f"{row['num_qudits']} | "
            f"{row['architecture']} | "
            f"{row['layout_passes']} | "
            f"{row['label']} | "
            f"{row.get('runtime_s') or ''} | "
            f"{row.get('can_exe_cum_s') or ''} | "
            f"{row.get('get_subgraph_cum_s') or ''} | "
            f"{row.get('can_exe_calls') or ''} | "
            f"{row.get('compiled_ops') or ''} |"
        )
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError('--repeats must be at least 1.')

    cases, skipped = build_cases(args)
    output_dir = make_output_dir(args)
    profile_dir = make_profile_dir(args, output_dir)

    labels = ('cg_legacy', 'cg_optimized', 'pgs_cg_compat')
    manifest = {
        'created_at': datetime.now().isoformat(),
        'python': sys.executable,
        'args': vars(args),
        'labels': labels,
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
    for repeat_index in range(args.repeats):
        for case_index, case in enumerate(cases):
            print(
                f'Repeat {repeat_index + 1}/{args.repeats}: '
                f'{case.circuit} on {case.architecture}, '
                f'{case.layout_passes} layout passes...'
            )
            for label in labels:
                profile_output = (
                    profile_dir
                    / (
                        f'repeat_{repeat_index + 1}__'
                        f'{case_index:04d}__{case_slug(case)}__{label}.prof'
                    )
                    if profile_dir is not None
                    else None
                )
                command = build_command(
                    case,
                    run_label_to_command_label(label),
                    track_heuristic_stats=args.track_heuristic_stats,
                    profile_output=profile_output,
                    cg_legacy_can_exe=label == 'cg_legacy',
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
                    repeat_index,
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

                row = {
                    'repeat': repeat_index + 1,
                    'circuit': case.circuit,
                    'num_qudits': case.num_qudits,
                    'architecture': case.architecture,
                    'layout_passes': case.layout_passes,
                    'label': label,
                    'status': status,
                    'runtime_s': metrics.get('runtime_s'),
                    'duration_s': duration_s,
                    'compiled_ops': metrics.get('compiled_ops'),
                    'initial_mapping': metrics.get('initial_mapping'),
                    'final_mapping': metrics.get('final_mapping'),
                    'placement': metrics.get('placement'),
                    **parse_profile_summary(profile_output),
                    'raw_output': raw_output,
                    'profile_output': None if profile_output is None else str(profile_output),
                    'command': ' '.join(command),
                }
                rows.append(row)
                run_records.append({
                    **row,
                    'return_code': None if completed is None else completed.returncode,
                    'timed_out': timed_out,
                })
                write_summary_csv(output_dir, rows)
                write_summary_md(output_dir, rows)
                (output_dir / 'runs.json').write_text(
                    json.dumps(run_records, indent=2),
                    encoding='utf-8',
                )

                if status != 'ok' and not args.continue_on_error:
                    raise SystemExit(
                        f'Run failed for {case.circuit} on {case.architecture} '
                        f'({label}, repeat={repeat_index + 1}, '
                        f'passes={case.layout_passes}).',
                    )

    print(f'Wrote ablation results to {output_dir}')


if __name__ == '__main__':
    main()
