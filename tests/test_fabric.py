import itertools
import time
from dataclasses import replace

import numpy as np
import pytest

from rollout_fabric.config import Config
from rollout_fabric.fabric import RunStats, run_async, run_sync
from rollout_fabric.pool import ThreadPool
from rollout_fabric.worker import Spec, WorkerCrashed, generate

QUIET = Config(
    workers=4, batch_size=4, steps=10, mean_latency_s=0.0, train_step_s=0.0, sandbox=False
)


class ScriptedPool:
    """Finishes a rollout after a scripted number of wait() calls, so a test fixes the order."""

    def __init__(self, cfg, delay=None, crash=()):
        self.cfg, self.delay, self.crash = cfg, delay or {}, set(crash)
        self.tick = 0
        self.jobs = {}
        self.submitted = []
        self.max_unfinished = 0
        self._handles = itertools.count()

    def submit(self, spec, version, logits):
        handle = next(self._handles)
        wait_calls = self.delay.get(spec.sample_id, 1) if spec.attempt == 0 else 1
        self.jobs[handle] = (spec, version, logits.copy(), self.tick + wait_calls)
        self.submitted.append(spec)
        self.max_unfinished = max(self.max_unfinished, len(self.jobs))
        return handle

    def wait(self, handles, timeout):
        self.tick += 1
        return [h for h in handles if self.jobs[h][3] <= self.tick]

    def result(self, handle):
        spec, version, logits, _ = self.jobs.pop(handle)
        if (spec.sample_id, spec.attempt) in self.crash:
            raise WorkerCrashed(f"scripted crash of {spec}")
        return generate(spec, version, logits, replace(self.cfg, crash_prob=0.0))

    def close(self):
        pass


class StuckPool(ScriptedPool):
    def wait(self, handles, timeout):
        time.sleep(0.02)
        return []


def threaded(cfg, fn, **kw):
    pool = ThreadPool(cfg)
    try:
        return fn(cfg, pool, **kw)
    finally:
        pool.close()


MODES = pytest.mark.parametrize("fn", [run_sync, run_async], ids=["sync", "async"])


# --- what every run guarantees ---------------------------------------------------------------


@MODES
def test_a_run_trains_exactly_steps_times_batch_size_samples(fn):
    s = fn(QUIET, ScriptedPool(QUIET))
    assert len(s.step_rewards) == QUIET.steps
    assert s.consumed == QUIET.steps * QUIET.batch_size == len(s.sample_ids) == len(s.staleness)


@MODES
def test_no_sample_is_trained_on_twice(fn):
    s = fn(QUIET, ScriptedPool(QUIET))
    assert len(set(s.sample_ids)) == len(s.sample_ids)


@MODES
def test_a_crashed_rollout_is_asked_again_and_nothing_is_lost(fn):
    pool = ScriptedPool(QUIET, crash={(3, 0), (3, 1), (10, 0)})
    s = fn(QUIET, pool)
    assert s.crashes == 3
    assert 3 in s.sample_ids and 10 in s.sample_ids
    assert Spec(3, 3, attempt=2) in pool.submitted  # the third try is the one that worked
    assert s.consumed == QUIET.steps * QUIET.batch_size
    assert len(set(s.sample_ids)) == len(s.sample_ids)


@MODES
def test_random_crashes_on_real_threads_lose_nothing(fn):
    cfg = replace(QUIET, crash_prob=0.3, steps=8)
    s = threaded(cfg, fn)
    assert s.crashes > 0
    assert s.consumed == cfg.steps * cfg.batch_size
    assert len(set(s.sample_ids)) == len(s.sample_ids)


@MODES
def test_the_callback_sees_every_step(fn):
    seen = []
    s = fn(QUIET, ScriptedPool(QUIET), on_step=seen.append)
    assert [e["step"] for e in seen] == list(range(1, QUIET.steps + 1))
    assert seen[-1]["consumed"] == s.consumed
    assert {e["mode"] for e in seen} == {s.mode}
    assert all(0.0 <= e["reward"] <= 1.0 for e in seen)


@MODES
def test_a_pool_that_never_answers_ends_in_a_timeout(fn):
    cfg = replace(QUIET, task_timeout_s=0.2)
    with pytest.raises(TimeoutError):
        fn(cfg, StuckPool(cfg))


@MODES
def test_the_same_seed_gives_the_same_learning_curve_when_the_order_is_the_same(fn):
    a = fn(QUIET, ScriptedPool(QUIET))
    b = fn(QUIET, ScriptedPool(QUIET))
    assert a.step_rewards == b.step_rewards and a.sample_ids == b.sample_ids


@MODES
def test_the_learner_improves_the_policy(fn):
    cfg = replace(QUIET, steps=40, batch_size=16)
    s = fn(cfg, ScriptedPool(cfg))
    assert np.mean(s.step_rewards[:3]) < 0.55
    assert s.final_reward() > 0.8


# --- sync ------------------------------------------------------------------------------------


def test_sync_never_uses_an_old_policy():
    s = run_sync(QUIET, ScriptedPool(QUIET, delay={0: 7, 5: 3}))
    assert set(s.staleness) == {0}


def test_sync_waits_for_the_slowest_sample_of_every_batch():
    pool = ScriptedPool(QUIET, delay={0: 9})
    run_sync(QUIET, pool)
    assert pool.tick >= 9 + (QUIET.steps - 1)  # step 1 alone costs the straggler's 9 waits


# --- async -----------------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0, 1, 2, 4])
def test_async_never_trains_on_a_sample_older_than_the_bound(eta):
    cfg = replace(QUIET, max_staleness=eta, steps=20, mean_latency_s=0.005, latency_sigma=1.5)
    s = threaded(cfg, run_async)
    assert max(s.staleness) <= eta
    assert s.consumed == cfg.steps * cfg.batch_size


def test_a_straggler_that_comes_back_too_late_is_dropped_not_used():
    cfg = replace(QUIET, max_staleness=1, steps=10)
    pool = ScriptedPool(cfg, delay={0: 6})
    s = run_async(cfg, pool)
    assert s.dropped_stale == 1
    assert 0 not in s.sample_ids
    assert max(s.staleness) == 1  # others were used while one step behind, never two


def test_a_sample_one_step_behind_is_kept_when_the_bound_allows_it():
    cfg = replace(QUIET, max_staleness=1, steps=10)
    s = run_async(cfg, ScriptedPool(cfg))
    assert s.dropped_stale == 0
    assert 1 in s.staleness


def test_with_bound_zero_async_behaves_like_sync():
    cfg = replace(QUIET, max_staleness=0)
    s = run_async(cfg, ScriptedPool(cfg, delay={2: 4}))
    assert set(s.staleness) == {0}
    assert s.consumed == cfg.steps * cfg.batch_size


def test_dispatch_never_runs_further_ahead_than_the_bound_allows():
    cfg = replace(QUIET, max_staleness=2, steps=15, batch_size=4)
    cap = (cfg.max_staleness + 1) * cfg.batch_size
    pool = ScriptedPool(cfg)  # every rollout is back at once, faster than the learner can train
    ahead = []
    s = run_async(
        cfg,
        pool,
        on_step=lambda e: ahead.append(len(pool.submitted) - e["consumed"] - e["dropped_stale"]),
    )
    assert ahead and max(ahead) <= cap
    assert s.consumed == cfg.steps * cfg.batch_size


def test_at_most_workers_times_inflight_rollouts_are_queued():
    cfg = replace(QUIET, workers=3, inflight_per_worker=2, max_staleness=4)
    pool = ScriptedPool(cfg, delay={i: 1 + i % 4 for i in range(300)})
    run_async(cfg, pool)
    assert pool.max_unfinished <= cfg.workers * cfg.inflight_per_worker


def test_the_oldest_samples_are_used_first():
    cfg = replace(QUIET, max_staleness=3, steps=6)
    s = run_async(cfg, ScriptedPool(cfg, delay={i: 1 + (i * 5) % 4 for i in range(300)}))
    # within one batch the policy version of the samples never goes down
    per_step = [s.staleness[i : i + cfg.batch_size] for i in range(0, s.consumed, cfg.batch_size)]
    assert all(row == sorted(row, reverse=True) for row in per_step)


# --- async against sync on real threads --------------------------------------------------------


def test_async_beats_sync_when_some_workers_are_slow():
    cfg = Config(
        workers=8,
        batch_size=16,
        steps=10,
        mean_latency_s=0.03,
        latency_sigma=1.5,
        train_step_s=0.01,
        sandbox=False,
    )
    sync = threaded(cfg, run_sync)
    asyn = threaded(cfg, run_async)
    assert asyn.samples_per_s > 1.3 * sync.samples_per_s


def test_the_learner_waits_less_in_async():
    cfg = Config(
        workers=8,
        batch_size=16,
        steps=10,
        mean_latency_s=0.03,
        latency_sigma=1.5,
        train_step_s=0.01,
        sandbox=False,
    )
    assert threaded(cfg, run_async).learner_idle < threaded(cfg, run_sync).learner_idle


# --- the numbers in RunStats -------------------------------------------------------------------


def test_time_to_reward_is_the_end_of_the_first_window_that_reaches_the_target():
    s = RunStats("x", step_rewards=[0.1, 0.5, 0.95, 0.95, 0.95], step_end_s=[1, 2, 3, 4, 5])
    assert s.time_to_reward(0.9, window=3) == 5
    assert s.time_to_reward(0.9, window=1) == 3
    assert s.time_to_reward(0.99) is None


def test_derived_numbers():
    s = RunStats(
        "x",
        wall_s=10.0,
        learner_busy_s=2.5,
        consumed=100,
        staleness=[0] * 90 + [4] * 10,
        step_rewards=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
    )
    assert s.samples_per_s == 10.0
    assert s.learner_idle == 0.75
    assert s.mean_staleness == pytest.approx(0.4)
    assert s.p95_staleness == 4.0
    assert s.final_reward(window=2) == 1.0
    assert s.final_reward() == pytest.approx(0.76)


def test_an_empty_run_has_zeros_not_errors():
    s = RunStats("x")
    assert (s.samples_per_s, s.learner_idle, s.mean_staleness, s.p95_staleness) == (0, 0, 0, 0)
    assert s.final_reward() == 0.0 and s.time_to_reward(0.5) is None
