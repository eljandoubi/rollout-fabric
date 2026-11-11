"""What a worker does for one rollout: pick a candidate, take the time to generate it, score it."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rollout_fabric.config import Config
from rollout_fabric.policy import Logits, Sample, sample_choice
from rollout_fabric.sandbox import Verdict, run_candidate
from rollout_fabric.tasks import TASKS


class WorkerCrashed(Exception):
    """The worker died before it returned a rollout (a real crash or an injected one)."""


_LABEL_REASON = {
    "correct": "ok",
    "wrong": "wrong_answer",
    "raises": "error",
    "syntax": "error",
    "hangs": "timeout",
}


@dataclass(frozen=True)
class Spec:
    """A rollout to produce. `attempt` grows each time a crash makes us ask again."""

    sample_id: int
    task: int
    attempt: int = 0


@dataclass(frozen=True)
class Rollout:
    sample: Sample
    reason: str  # sandbox verdict: ok, wrong_answer, error, timeout, ...
    generate_s: float
    score_s: float


def draw_latency(rng: np.random.Generator, cfg: Config) -> float:
    """Log-normal generation time with mean `cfg.mean_latency_s`."""
    if cfg.mean_latency_s <= 0:
        return 0.0
    mu = np.log(cfg.mean_latency_s) - cfg.latency_sigma**2 / 2
    return float(rng.lognormal(mu, cfg.latency_sigma))


def generate(spec: Spec, version: int, logits: Logits, cfg: Config) -> Rollout:
    """The randomness depends on (seed, sample, attempt) only, never on which worker runs it."""
    rng = np.random.default_rng([cfg.seed, spec.sample_id, spec.attempt])
    if cfg.crash_prob > 0 and rng.random() < cfg.crash_prob:
        raise WorkerCrashed(f"sample {spec.sample_id}, attempt {spec.attempt}")
    choice, logp = sample_choice(logits, spec.task, rng)
    generate_s = draw_latency(rng, cfg)
    time.sleep(generate_s)
    task = TASKS[spec.task]
    if cfg.sandbox:
        verdict = run_candidate(
            task.candidates[choice],
            task.entry,
            task.inputs,
            task.expected,
            timeout_s=cfg.sandbox_timeout_s,
        )
    else:
        # trust the label of the candidate: same reasons as the sandbox, no process to start
        kind = task.kinds[choice]
        verdict = Verdict(kind == "correct", _LABEL_REASON[kind], 0.0)
    reward = 1.0 if verdict.passed else 0.0
    sample = Sample(spec.sample_id, spec.task, choice, reward, logp, version)
    return Rollout(sample, verdict.reason, generate_s, verdict.wall_s)
