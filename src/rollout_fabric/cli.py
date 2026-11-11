"""Command line: `run` for one run, `compare` for sync against async."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from rollout_fabric.config import Config
from rollout_fabric.fabric import OnStep, RunStats, run_async, run_sync
from rollout_fabric.metrics import dogstatsd_sink, fanout, jsonl_sink, prometheus_text
from rollout_fabric.pool import make_pool, runtime
from rollout_fabric.report import Summary, comparison, markdown_table, median_summary, summarize


def _quick(cfg: Config) -> Config:
    """Shorter timings, for smoke tests."""
    return replace(cfg, steps=12, mean_latency_s=0.02, train_step_s=0.01)


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    """One option per Config field, so the CLI cannot drift from the settings."""
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            kind = type(f.default)
            parser.add_argument(flag, type=kind, default=f.default, help=f"default {f.default}")


def _config(args: argparse.Namespace) -> Config:
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    return _quick(cfg) if getattr(args, "quick", False) else cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rollout-fabric", description="Sync against async rollout generation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="one run")
    run.add_argument("--mode", choices=["sync", "async"], default="async")
    run.add_argument("--backend", choices=["thread", "ray"], default="thread")
    run.add_argument("--target", type=float, default=0.9, help="reward to time (default 0.9)")
    run.add_argument("--jsonl", type=Path, help="append one JSON line per learner step")
    run.add_argument("--statsd", metavar="HOST:PORT", help="send step gauges as DogStatsD")
    run.add_argument("--prom-file", type=Path, help="write the final numbers in Prometheus format")
    _add_config_options(run)

    cmp = sub.add_parser("compare", help="sync and async on the same settings")
    cmp.add_argument("--backend", choices=["thread", "ray"], default="thread")
    cmp.add_argument("--target", type=float, default=0.9, help="reward to time (default 0.9)")
    cmp.add_argument("--repeat", type=int, default=1, help="runs per mode, seeds seed..seed+N-1")
    cmp.add_argument("--quick", action="store_true", help="shorter timings, for smoke tests")
    _add_config_options(cmp)
    return parser


def _run_once(mode: str, cfg: Config, backend: str, on_step: OnStep | None = None) -> RunStats:
    pool = make_pool(backend, cfg)
    try:
        controller = run_sync if mode == "sync" else run_async
        return controller(cfg, pool, on_step=on_step)
    finally:
        pool.close()


def _statsd_sink(address: str) -> OnStep:
    host, _, port = address.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"--statsd wants HOST:PORT, got {address!r}")
    return dogstatsd_sink(host, int(port))


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    sinks: list[OnStep] = []
    if args.jsonl:
        sinks.append(jsonl_sink(args.jsonl))
    if args.statsd:
        sinks.append(_statsd_sink(args.statsd))
    with runtime(args.backend):
        stats = _run_once(args.mode, cfg, args.backend, fanout(sinks))
    if args.prom_file:
        args.prom_file.write_text(prometheus_text(stats, args.target), encoding="utf-8")
    print(markdown_table({args.mode: summarize(stats, args.target)}, args.target))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = _config(args)
    results: dict[str, list[Summary]] = {"sync": [], "async": []}
    with runtime(args.backend):
        for k in range(args.repeat):
            for mode in results:
                stats = _run_once(mode, replace(cfg, seed=cfg.seed + k), args.backend)
                results[mode].append(summarize(stats, args.target))
    rows = {mode: median_summary(items) for mode, items in results.items()}
    print(markdown_table(rows, args.target))
    print()
    print(comparison(rows["sync"], rows["async"], args.target))
    if args.repeat > 1:
        print(f"(median of {args.repeat} runs per mode)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _cmd_run(args) if args.command == "run" else _cmd_compare(args)
