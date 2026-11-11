"""Turns the statistics of runs into the table a person reads."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass, fields

from rollout_fabric.fabric import RunStats


@dataclass(frozen=True)
class Summary:
    wall_s: float
    samples_per_s: float
    learner_idle: float
    mean_staleness: float
    p95_staleness: float
    dropped_stale: float
    crashes: float
    to_target_s: float | None  # None: the target reward was never reached
    final_reward: float
    reached: int = 1  # how many of the `runs` reached the target
    runs: int = 1


def summarize(stats: RunStats, target: float) -> Summary:
    to_target = stats.time_to_reward(target)
    return Summary(
        wall_s=stats.wall_s,
        samples_per_s=stats.samples_per_s,
        learner_idle=stats.learner_idle,
        mean_staleness=stats.mean_staleness,
        p95_staleness=stats.p95_staleness,
        dropped_stale=stats.dropped_stale,
        crashes=stats.crashes,
        to_target_s=to_target,
        final_reward=stats.final_reward(),
        reached=int(to_target is not None),
    )


def median_summary(items: list[Summary]) -> Summary:
    """Median of every column over repeated runs. The time to target is the median of the runs
    that got there, and `reached` says how many did."""
    if not items:
        raise ValueError("no runs to summarize")
    medians = {
        f.name: statistics.median(getattr(s, f.name) for s in items)
        for f in fields(Summary)
        if f.name not in ("to_target_s", "reached", "runs")
    }
    times = [s.to_target_s for s in items if s.to_target_s is not None]
    return Summary(
        **medians,
        to_target_s=statistics.median(times) if times else None,
        reached=len(times),
        runs=len(items),
    )


def _time_to_target(s: Summary) -> str:
    if s.to_target_s is None:
        return "not reached"
    return f"{s.to_target_s:.1f}" + ("" if s.reached == s.runs else f" ({s.reached}/{s.runs} runs)")


def markdown_table(rows: Mapping[str, Summary], target: float) -> str:
    header = (
        f"| mode | wall (s) | samples/s | learner idle | staleness mean / p95 | dropped | crashes "
        f"| seconds to reward {target:g} | final reward |"
    )
    rule = "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"
    lines = [header, rule]
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s.wall_s:.1f} | {s.samples_per_s:.1f} | {s.learner_idle:.0%} "
            f"| {s.mean_staleness:.2f} / {s.p95_staleness:g} | {s.dropped_stale:g} "
            f"| {s.crashes:g} | {_time_to_target(s)} | {s.final_reward:.2f} |"
        )
    return "\n".join(lines)


def comparison(sync: Summary, asyn: Summary, target: float) -> str:
    """One sentence on what async bought, with the numbers behind it."""
    parts = [f"async vs sync: {asyn.samples_per_s / sync.samples_per_s:.2f}x samples/s"]
    parts.append(f"learner idle {sync.learner_idle:.0%} -> {asyn.learner_idle:.0%}")
    if sync.to_target_s and asyn.to_target_s:
        parts.append(
            f"{sync.to_target_s / asyn.to_target_s:.2f}x faster to reward {target:g} "
            f"({sync.to_target_s:.1f} s -> {asyn.to_target_s:.1f} s)"
        )
    return ", ".join(parts)
