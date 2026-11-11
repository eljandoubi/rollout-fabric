"""Settings shared by the workers and the controllers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    workers: int = 4
    batch_size: int = 16  # rollouts per learner step
    steps: int = 30  # learner steps to run
    max_staleness: int = 2  # async only: oldest policy version a sample may come from, in steps
    inflight_per_worker: int = 2  # async only: rollouts queued per worker
    mean_latency_s: float = 0.05  # mean time a worker needs to generate one completion
    latency_sigma: float = 1.0  # log-normal spread: the tail is what makes stragglers
    train_step_s: float = 0.03  # time the learner needs for one update
    lr: float = 0.5
    seed: int = 0
    crash_prob: float = 0.0  # chance that a worker dies while handling one rollout
    sandbox: bool = True  # run the candidate in the sandbox, or trust its label
    sandbox_timeout_s: float = 0.5
    task_timeout_s: float = 60.0  # give up when nothing finishes for this long
