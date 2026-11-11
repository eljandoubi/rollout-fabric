"""Where the numbers go: a JSONL file, DogStatsD over UDP, a Prometheus text file."""

from __future__ import annotations

import contextlib
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rollout_fabric.fabric import OnStep, RunStats

GAUGES = ("reward", "mean_staleness", "dropped_stale", "crashes", "consumed", "elapsed_s")


def fanout(sinks: list[OnStep]) -> OnStep | None:
    """One callback that feeds every sink, or None when there is nothing to feed."""
    if not sinks:
        return None

    def emit(event: dict[str, Any]) -> None:
        for sink in sinks:
            sink(event)

    return emit


def jsonl_sink(path: Path) -> OnStep:
    """Appends one JSON object per learner step."""

    def emit(event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    return emit


def dogstatsd_line(prefix: str, name: str, value: float, tags: list[str]) -> str:
    """`rollout_fabric.reward:0.5|g|#mode:async`, the DogStatsD gauge format."""
    line = f"{prefix}.{name}:{value:g}|g"
    return line + ("|#" + ",".join(tags) if tags else "")


def dogstatsd_sink(
    host: str = "127.0.0.1", port: int = 8125, prefix: str = "rollout_fabric"
) -> Callable[[dict[str, Any]], None]:
    """Sends the step gauges as one UDP datagram per step, fire and forget like the agent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit(event: dict[str, Any]) -> None:
        tags = [f"mode:{event['mode']}"]
        lines = [dogstatsd_line(prefix, name, float(event[name]), tags) for name in GAUGES]
        # a metrics outage must never stop training
        with contextlib.suppress(OSError):
            sock.sendto("\n".join(lines).encode(), (host, port))

    return emit


def prometheus_text(stats: RunStats, target: float = 0.9) -> str:
    """The final numbers of a run in the Prometheus text format (for the textfile collector)."""
    mode = f'mode="{stats.mode}"'
    reached = stats.time_to_reward(target)
    metrics: list[tuple[str, str, str, float]] = [
        ("wall_seconds", "gauge", "Wall time of the run.", stats.wall_s),
        ("samples_consumed_total", "counter", "Samples the learner trained on.", stats.consumed),
        (
            "samples_dropped_stale_total",
            "counter",
            "Samples dropped for being too old.",
            stats.dropped_stale,
        ),
        ("worker_crashes_total", "counter", "Rollouts lost to a dead worker.", stats.crashes),
        ("samples_per_second", "gauge", "Consumed samples per second.", stats.samples_per_s),
        ("learner_idle_ratio", "gauge", "Share of the run the learner waited.", stats.learner_idle),
        ("staleness_mean", "gauge", "Mean policy lag of consumed samples.", stats.mean_staleness),
        ("staleness_p95", "gauge", "95th percentile policy lag.", stats.p95_staleness),
        ("final_reward", "gauge", "Mean reward of the last batches.", stats.final_reward()),
    ]
    if reached is not None:
        metrics.append(
            ("seconds_to_target_reward", "gauge", f"Seconds to reach reward {target}.", reached)
        )
    lines = []
    for name, kind, help_text, value in metrics:
        lines += [
            f"# HELP rollout_fabric_{name} {help_text}",
            f"# TYPE rollout_fabric_{name} {kind}",
            f"rollout_fabric_{name}{{{mode}}} {value:g}",
        ]
    lines += [
        "# HELP rollout_fabric_verdicts_total Sandbox verdicts of consumed samples.",
        "# TYPE rollout_fabric_verdicts_total counter",
    ]
    for reason, count in sorted(stats.reasons.items()):
        lines.append(f'rollout_fabric_verdicts_total{{{mode},reason="{reason}"}} {count}')
    return "\n".join(lines) + "\n"
