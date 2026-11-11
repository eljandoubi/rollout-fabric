"""Two ways to drive the workers: wait for the whole batch (sync), or keep every worker busy
and train on whatever is fresh enough (async)."""

from __future__ import annotations

import itertools
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from rollout_fabric.config import Config
from rollout_fabric.policy import Policy
from rollout_fabric.pool import Handle, Pool
from rollout_fabric.tasks import N_CHOICES, TASKS
from rollout_fabric.worker import Rollout, Spec, WorkerCrashed

OnStep = Callable[[dict[str, Any]], None]


@dataclass
class RunStats:
    mode: str
    wall_s: float = 0.0
    learner_busy_s: float = 0.0
    consumed: int = 0
    dropped_stale: int = 0
    crashes: int = 0
    staleness: list[int] = field(default_factory=list)  # one entry per consumed sample
    sample_ids: list[int] = field(default_factory=list)  # the same samples, in training order
    step_rewards: list[float] = field(default_factory=list)  # mean reward of each batch
    step_end_s: list[float] = field(default_factory=list)  # when each step ended
    reasons: Counter[str] = field(default_factory=Counter)  # sandbox verdicts of consumed samples

    @property
    def samples_per_s(self) -> float:
        return self.consumed / self.wall_s if self.wall_s else 0.0

    @property
    def learner_idle(self) -> float:
        """Share of the run during which the learner had nothing to train on."""
        return 1.0 - self.learner_busy_s / self.wall_s if self.wall_s else 0.0

    @property
    def mean_staleness(self) -> float:
        return float(np.mean(self.staleness)) if self.staleness else 0.0

    @property
    def p95_staleness(self) -> float:
        return float(np.percentile(self.staleness, 95)) if self.staleness else 0.0

    def final_reward(self, window: int = 5) -> float:
        return float(np.mean(self.step_rewards[-window:])) if self.step_rewards else 0.0

    def time_to_reward(self, target: float, window: int = 3) -> float | None:
        """Seconds until the mean reward of `window` consecutive batches reaches `target`."""
        for i in range(window - 1, len(self.step_rewards)):
            if np.mean(self.step_rewards[i - window + 1 : i + 1]) >= target:
                return self.step_end_s[i]
        return None


class _Run:
    """What both controllers share: the policy, the counters, how a step is trained."""

    def __init__(self, mode: str, cfg: Config, on_step: OnStep | None) -> None:
        self.cfg = cfg
        self.on_step = on_step
        self.policy = Policy(len(TASKS), N_CHOICES, lr=cfg.lr)
        self.stats = RunStats(mode)
        self.t0 = time.monotonic()
        self.last_progress = self.t0
        self._ids = itertools.count()

    def new_spec(self) -> Spec:
        i = next(self._ids)
        return Spec(i, i % len(TASKS))

    def submit(self, pool: Pool, spec: Spec) -> Handle:
        return pool.submit(spec, self.policy.version, self.policy.logits)

    def collect(self, pool: Pool, pending: dict[Handle, Spec], done: list[Handle]) -> list[Rollout]:
        """Rollouts of the finished handles. A crashed one is asked again with a new attempt."""
        rollouts = []
        for handle in done:
            spec = pending.pop(handle)
            try:
                rollouts.append(pool.result(handle))
            except WorkerCrashed as crash:
                self.stats.crashes += 1
                if spec.attempt + 1 >= self.cfg.max_attempts:
                    raise RuntimeError(
                        f"sample {spec.sample_id} killed its worker {self.cfg.max_attempts} times"
                    ) from crash
                again = replace(spec, attempt=spec.attempt + 1)
                pending[self.submit(pool, again)] = again
        if done:
            self.last_progress = time.monotonic()
        return rollouts

    def check_stall(self) -> None:
        if time.monotonic() - self.last_progress > self.cfg.task_timeout_s:
            raise TimeoutError(f"no rollout finished for {self.cfg.task_timeout_s} s")

    def train(self, batch: list[Rollout]) -> None:
        started = time.monotonic()
        for r in batch:
            self.stats.staleness.append(self.policy.version - r.sample.version)
            self.stats.sample_ids.append(r.sample.sample_id)
            self.stats.reasons[r.reason] += 1
        self.policy.update([r.sample for r in batch])
        time.sleep(self.cfg.train_step_s)  # the learner is busy; the workers keep going
        now = time.monotonic()
        stats = self.stats
        stats.learner_busy_s += now - started
        stats.consumed += len(batch)
        stats.step_rewards.append(float(np.mean([r.sample.reward for r in batch])))
        stats.step_end_s.append(now - self.t0)
        self.last_progress = now
        if self.on_step:
            self.on_step(
                {
                    "mode": stats.mode,
                    "step": self.policy.version,
                    "reward": stats.step_rewards[-1],
                    "consumed": stats.consumed,
                    "dropped_stale": stats.dropped_stale,
                    "crashes": stats.crashes,
                    "mean_staleness": float(np.mean(stats.staleness[-len(batch) :])),
                    "elapsed_s": stats.step_end_s[-1],
                    "learner_busy_s": stats.learner_busy_s,
                }
            )

    def finish(self) -> RunStats:
        self.stats.wall_s = time.monotonic() - self.t0
        return self.stats


def run_sync(cfg: Config, pool: Pool, on_step: OnStep | None = None) -> RunStats:
    """Every step waits for its whole batch, so one slow worker stalls everybody."""
    run = _Run("sync", cfg, on_step)
    for _ in range(cfg.steps):
        pending: dict[Handle, Spec] = {}
        for _ in range(cfg.batch_size):
            spec = run.new_spec()
            pending[run.submit(pool, spec)] = spec
        batch: list[Rollout] = []
        while pending:
            done = pool.wait(list(pending), timeout=0.5)
            batch += run.collect(pool, pending, done)
            run.check_stall()
        run.train(batch)
    return run.finish()


def run_async(cfg: Config, pool: Pool, on_step: OnStep | None = None) -> RunStats:
    """Workers never wait for the learner.

    Dispatch stops when `dispatched - consumed - dropped` reaches (max_staleness + 1) batches,
    so we never run further ahead than the staleness bound allows. A sample that is still older
    than `max_staleness` steps when its turn comes is dropped, never trained on.
    """
    run = _Run("async", cfg, on_step)
    cap = (cfg.max_staleness + 1) * cfg.batch_size
    max_inflight = cfg.workers * cfg.inflight_per_worker
    inflight: dict[Handle, Spec] = {}
    buffer: list[Rollout] = []
    while run.policy.version < cfg.steps:
        while len(inflight) < max_inflight and len(inflight) + len(buffer) < cap:
            spec = run.new_spec()
            inflight[run.submit(pool, spec)] = spec
        done = pool.wait(list(inflight), timeout=0.05) if inflight else []
        buffer += run.collect(pool, inflight, done)
        fresh = [r for r in buffer if run.policy.version - r.sample.version <= cfg.max_staleness]
        run.stats.dropped_stale += len(buffer) - len(fresh)
        buffer = fresh
        if len(buffer) >= cfg.batch_size:
            buffer.sort(key=lambda r: (r.sample.version, r.sample.sample_id))  # oldest first
            batch, buffer = buffer[: cfg.batch_size], buffer[cfg.batch_size :]
            run.train(batch)
        run.check_stall()
    return run.finish()
