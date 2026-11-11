from dataclasses import replace

import numpy as np
import pytest

from rollout_fabric.config import Config
from rollout_fabric.policy import softmax
from rollout_fabric.tasks import N_CHOICES, TASKS
from rollout_fabric.worker import Spec, WorkerCrashed, draw_latency, generate

FAST = Config(mean_latency_s=0.0, sandbox=False)


def flat() -> np.ndarray:
    return np.zeros((len(TASKS), N_CHOICES))


def logits_for(task: int, choice: int) -> np.ndarray:
    """A policy that picks `choice` for `task` almost surely."""
    logits = flat()
    logits[task, choice] = 50.0
    return logits


def test_the_same_spec_gives_the_same_rollout_whoever_runs_it():
    a = generate(Spec(7, 2), 3, flat(), FAST)
    b = generate(Spec(7, 2), 3, flat(), FAST)
    assert a == b


def test_another_attempt_draws_again():
    picks = {generate(Spec(7, 2, attempt), 0, flat(), FAST).sample.choice for attempt in range(30)}
    assert len(picks) > 1


def test_another_seed_draws_again():
    picks = {
        generate(Spec(7, 2), 0, flat(), replace(FAST, seed=s)).sample.choice for s in range(30)
    }
    assert len(picks) > 1


def test_the_sample_remembers_the_version_and_the_log_probability():
    logits = flat()
    r = generate(Spec(1, 0), 5, logits, FAST)
    assert r.sample.version == 5 and r.sample.sample_id == 1 and r.sample.task == 0
    assert r.sample.behavior_logp == pytest.approx(np.log(softmax(logits[0])[r.sample.choice]))


@pytest.mark.parametrize("task", range(len(TASKS)))
def test_without_the_sandbox_the_label_of_the_candidate_decides(task):
    for choice, kind in enumerate(TASKS[task].kinds):
        r = generate(Spec(0, task), 0, logits_for(task, choice), FAST)
        assert (r.sample.reward == 1.0) == (kind == "correct")


def test_the_unsandboxed_reasons_match_the_sandbox_reasons():
    reasons = {
        kind: generate(Spec(0, 0), 0, logits_for(0, i), FAST).reason
        for i, kind in enumerate(TASKS[0].kinds)
    }
    assert reasons == {
        "correct": "ok",
        "wrong": "wrong_answer",
        "raises": "error",
        "hangs": "timeout",
        "syntax": "error",
    }


def test_with_the_sandbox_a_correct_candidate_is_rewarded_and_a_wrong_one_is_not():
    cfg = replace(FAST, sandbox=True)
    good = TASKS[1].kinds.index("correct")
    bad = TASKS[1].kinds.index("wrong")
    assert generate(Spec(0, 1), 0, logits_for(1, good), cfg).sample.reward == 1.0
    r = generate(Spec(0, 1), 0, logits_for(1, bad), cfg)
    assert r.sample.reward == 0.0 and r.reason == "wrong_answer"


def test_the_sandbox_catches_a_hanging_candidate():
    cfg = replace(FAST, sandbox=True, sandbox_timeout_s=0.3)
    hang = TASKS[0].kinds.index("hangs")
    r = generate(Spec(0, 0), 0, logits_for(0, hang), cfg)
    assert r.sample.reward == 0.0 and r.reason == "timeout"


def test_a_worker_with_crash_probability_one_always_crashes():
    cfg = replace(FAST, crash_prob=1.0)
    with pytest.raises(WorkerCrashed):
        generate(Spec(0, 0), 0, flat(), cfg)


def test_a_worker_with_crash_probability_zero_never_crashes():
    for i in range(50):
        generate(Spec(i, 0), 0, flat(), FAST)


def test_crashes_happen_at_about_the_requested_rate_and_a_retry_can_succeed():
    cfg = replace(FAST, crash_prob=0.3)
    crashed = 0
    for i in range(400):
        try:
            generate(Spec(i, 0), 0, flat(), cfg)
        except WorkerCrashed:
            crashed += 1
    assert 0.2 < crashed / 400 < 0.4
    # a spec that crashed on attempt 0 gets other draws on the next attempts
    first = next(i for i in range(400) if _crashes(Spec(i, 0), cfg))
    assert any(not _crashes(Spec(first, 0, attempt), cfg) for attempt in range(1, 20))


def _crashes(spec, cfg) -> bool:
    try:
        generate(spec, 0, flat(), cfg)
    except WorkerCrashed:
        return True
    return False


def test_latency_has_the_requested_mean_and_a_heavy_tail():
    cfg = Config(mean_latency_s=0.2, latency_sigma=1.0)
    rng = np.random.default_rng(0)
    draws = np.array([draw_latency(rng, cfg) for _ in range(40_000)])
    assert draws.mean() == pytest.approx(0.2, rel=0.05)
    assert np.quantile(draws, 0.99) > 4 * np.median(draws)


def test_zero_mean_latency_means_no_waiting():
    assert draw_latency(np.random.default_rng(0), Config(mean_latency_s=0.0)) == 0.0


def test_the_generation_time_is_spent():
    cfg = replace(FAST, mean_latency_s=0.05, latency_sigma=0.0)
    r = generate(Spec(0, 0), 0, flat(), cfg)
    assert r.generate_s == pytest.approx(0.05)
