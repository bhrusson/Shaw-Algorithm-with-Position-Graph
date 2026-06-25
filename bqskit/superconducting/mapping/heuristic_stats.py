"""Helpers for recording SABRE heuristic-work statistics."""
from __future__ import annotations

from typing import Any
from typing import Mapping


SUM_KEYS = (
    'best_swap_calls',
    'score_swap_calls',
    'frontier_size_sum',
    'extended_size_sum',
    'affected_frontier_sum',
    'affected_extend_sum',
    'affected_total_sum',
    'full_rescore_terms_sum',
)

MAX_KEYS = (
    'frontier_size_max',
    'extended_size_max',
    'candidate_count_max',
    'affected_frontier_max',
    'affected_extend_max',
    'affected_total_max',
    'full_rescore_terms_max',
)

RAW_KEYS = SUM_KEYS + MAX_KEYS


def new_heuristic_stats() -> dict[str, int]:
    """Return an empty raw heuristic-stats dictionary."""
    return {key: 0 for key in RAW_KEYS}


def ensure_heuristic_stats(owner: Any) -> dict[str, int]:
    """Return the owner's raw stats dictionary, creating it if needed."""
    stats = getattr(owner, '_heuristic_stats', None)
    if stats is None:
        stats = new_heuristic_stats()
        setattr(owner, '_heuristic_stats', stats)
    return stats


def reset_heuristic_stats(owner: Any) -> dict[str, int]:
    """Reset the owner's heuristic stats and return the fresh dictionary."""
    stats = new_heuristic_stats()
    setattr(owner, '_heuristic_stats', stats)
    return stats


def record_best_swap(
    stats: dict[str, int],
    frontier_size: int,
    extended_size: int,
    candidate_count: int,
) -> None:
    """Record one best-swap decision point."""
    stats['best_swap_calls'] += 1
    stats['score_swap_calls'] += candidate_count
    stats['frontier_size_sum'] += frontier_size
    stats['extended_size_sum'] += extended_size
    stats['frontier_size_max'] = max(stats['frontier_size_max'], frontier_size)
    stats['extended_size_max'] = max(stats['extended_size_max'], extended_size)
    stats['candidate_count_max'] = max(stats['candidate_count_max'], candidate_count)


def record_candidate(
    stats: dict[str, int],
    full_rescore_terms: int,
    affected_frontier: int,
    affected_extend: int,
) -> None:
    """Record one candidate evaluation."""
    affected_total = affected_frontier + affected_extend
    stats['affected_frontier_sum'] += affected_frontier
    stats['affected_extend_sum'] += affected_extend
    stats['affected_total_sum'] += affected_total
    stats['full_rescore_terms_sum'] += full_rescore_terms

    stats['affected_frontier_max'] = max(
        stats['affected_frontier_max'],
        affected_frontier,
    )
    stats['affected_extend_max'] = max(
        stats['affected_extend_max'],
        affected_extend,
    )
    stats['affected_total_max'] = max(
        stats['affected_total_max'],
        affected_total,
    )
    stats['full_rescore_terms_max'] = max(
        stats['full_rescore_terms_max'],
        full_rescore_terms,
    )


def summarize_heuristic_stats(
    stats: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """Return raw stats plus derived averages and ratios."""
    raw = new_heuristic_stats()
    if stats is not None:
        for key in RAW_KEYS:
            raw[key] = int(stats.get(key, 0) or 0)

    best_swap_calls = raw['best_swap_calls']
    score_swap_calls = raw['score_swap_calls']
    affected_total_sum = raw['affected_total_sum']
    full_rescore_terms_sum = raw['full_rescore_terms_sum']

    def avg(total: int, count: int) -> float | None:
        return (total / count) if count else None

    summary: dict[str, int | float | None] = dict(raw)
    summary.update({
        'frontier_size_avg_per_best_swap': avg(
            raw['frontier_size_sum'],
            best_swap_calls,
        ),
        'extended_size_avg_per_best_swap': avg(
            raw['extended_size_sum'],
            best_swap_calls,
        ),
        'candidate_count_avg_per_best_swap': avg(
            score_swap_calls,
            best_swap_calls,
        ),
        'affected_frontier_avg_per_candidate': avg(
            raw['affected_frontier_sum'],
            score_swap_calls,
        ),
        'affected_extend_avg_per_candidate': avg(
            raw['affected_extend_sum'],
            score_swap_calls,
        ),
        'affected_total_avg_per_candidate': avg(
            affected_total_sum,
            score_swap_calls,
        ),
        'full_rescore_terms_avg_per_candidate': avg(
            full_rescore_terms_sum,
            score_swap_calls,
        ),
        'rescore_terms_ratio': (
            full_rescore_terms_sum / affected_total_sum
            if affected_total_sum
            else None
        ),
        'affected_fraction_of_full_rescore': (
            affected_total_sum / full_rescore_terms_sum
            if full_rescore_terms_sum
            else None
        ),
    })
    return summary


def combine_heuristic_stats(
    *stats_items: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """Combine multiple raw-or-summarized stats dictionaries."""
    combined = new_heuristic_stats()
    for stats in stats_items:
        if stats is None:
            continue
        for key in SUM_KEYS:
            combined[key] += int(stats.get(key, 0) or 0)
        for key in MAX_KEYS:
            combined[key] = max(combined[key], int(stats.get(key, 0) or 0))

    return summarize_heuristic_stats(combined)
