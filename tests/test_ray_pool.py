"""Behaviour that only Ray has: real processes that die and come back."""

import time
from dataclasses import replace

import numpy as np
import pytest

from rollout_fabric.config import Config
from rollout_fabric.fabric import run_async, run_sync
from rollout_fabric.ray_pool import RayPool
from rollout_fabric.tasks import N_CHOICES, TASKS
from rollout_fabric.worker import Spec, WorkerCrashed

pytestmark = pytest.mark.ray

FAST = Config(workers=2, mean_latency_s=0.0, sandbox=False)


def flat() -> np.ndarray:
    return np.zeros((len(TASKS), N_CHOICES))


@pytest.fixture
def pool_of(ray_session):
    pools = []

    def make(cfg):
        pools.append(RayPool(cfg))
        return pools[-1]

    yield make
    for pool in pools:
        pool.close()


def finish(pool, handles, timeout=60.0):
    """Wait for every handle; returns {handle: rollout or the WorkerCrashed it ended with}."""
    pending, out = list(handles), {}
    deadline = time.monotonic() + timeout
    while pending:
        assert time.monotonic() < deadline, "handles did not finish"
        for h in pool.wait(pending, timeout=0.2):
            pending.remove(h)
            try:
                out[h] = pool.result(h)
            except WorkerCrashed as e:
                out[h] = e
    return out


def test_a_rollout_waits_in_the_pool_until_a_worker_is_free(pool_of):
    pool = pool_of(replace(FAST, workers=2, mean_latency_s=0.3, latency_sigma=0.0))
    started = time.monotonic()
    jobs = [pool.submit(Spec(i, 0), 0, flat()) for i in range(5)]
    assert sum(j.ref is not None for j in jobs) == 2  # the other three wait their turn here
    assert all(not isinstance(r, Exception) for r in finish(pool, jobs).values())
    assert 0.8 < time.monotonic() - started < 3.0  # three rounds of 0.3 s


def test_a_crash_kills_the_worker_process_and_ray_brings_it_back(pool_of, ray_session):
    pool = pool_of(replace(FAST, workers=1, crash_prob=1.0))
    before = ray_session.get(pool._actors[0].ping.remote())
    job = pool.submit(Spec(0, 0), 0, flat())
    assert isinstance(finish(pool, [job])[job], WorkerCrashed)
    after = ray_session.get(pool._actors[0].ping.remote())  # served by the restarted process
    assert after != before


def test_a_crash_loses_only_the_rollout_that_was_running(pool_of):
    cfg = replace(FAST, workers=2, crash_prob=0.5)
    pool = pool_of(cfg)
    doomed = {i for i in range(16) if np.random.default_rng([cfg.seed, i, 0]).random() < 0.5}
    assert 3 < len(doomed) < 13  # the seed gives a real mix of crashes and successes
    jobs = {pool.submit(Spec(i, i % len(TASKS)), 0, flat()): i for i in range(16)}
    results = finish(pool, list(jobs))
    crashed = {jobs[h] for h, r in results.items() if isinstance(r, WorkerCrashed)}
    assert crashed == doomed
    ok = {jobs[h]: r for h, r in results.items() if not isinstance(r, WorkerCrashed)}
    assert all(r.sample.sample_id == i for i, r in ok.items())


@pytest.mark.parametrize("fn", [run_sync, run_async], ids=["sync", "async"])
def test_a_full_run_survives_worker_crashes(pool_of, fn):
    cfg = replace(FAST, workers=3, batch_size=6, steps=5, mean_latency_s=0.01, crash_prob=0.25)
    pool = pool_of(cfg)
    s = fn(cfg, pool)
    assert s.crashes > 0
    assert s.consumed == cfg.steps * cfg.batch_size
    assert len(set(s.sample_ids)) == len(s.sample_ids)
    assert max(s.staleness) <= cfg.max_staleness


def test_ray_and_threads_learn_the_same_curve_in_sync_mode(pool_of):
    cfg = replace(FAST, workers=3, batch_size=12, steps=8, lr=3.0, seed=4)
    on_ray = run_sync(cfg, pool_of(cfg))
    from rollout_fabric.pool import ThreadPool

    threads = ThreadPool(cfg)
    try:
        on_threads = run_sync(cfg, threads)
    finally:
        threads.close()
    assert on_ray.step_rewards == on_threads.step_rewards
    assert on_ray.reasons == on_threads.reasons


def test_async_on_ray_beats_sync_when_some_workers_are_slow(pool_of):
    cfg = Config(
        workers=4,
        batch_size=8,
        steps=8,
        mean_latency_s=0.05,
        latency_sigma=1.5,
        train_step_s=0.02,
        sandbox=False,
        seed=1,
    )
    sync = run_sync(cfg, pool_of(cfg))
    asyn = run_async(cfg, pool_of(cfg))
    assert asyn.samples_per_s > 1.2 * sync.samples_per_s
