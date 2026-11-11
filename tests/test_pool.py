import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import pytest

from rollout_fabric.config import Config
from rollout_fabric.pool import ThreadPool, make_pool, runtime
from rollout_fabric.tasks import N_CHOICES, TASKS
from rollout_fabric.worker import Spec, WorkerCrashed

FAST = Config(workers=2, mean_latency_s=0.0, sandbox=False)


def flat() -> np.ndarray:
    return np.zeros((len(TASKS), N_CHOICES))


def drain(pool, handles, timeout=20.0):
    """Wait until every handle has finished."""
    pending, done = list(handles), []
    deadline = time.monotonic() + timeout
    while pending:
        assert time.monotonic() < deadline, "handles did not finish"
        for h in pool.wait(pending, timeout=0.1):
            pending.remove(h)
            done.append(h)
    return done


# These run against every backend (threads, and Ray when it is installed).


def test_a_submitted_rollout_comes_back(new_pool):
    pool = new_pool(FAST)
    h = pool.submit(Spec(3, 1), 4, flat())
    assert drain(pool, [h]) == [h]
    r = pool.result(h)
    assert r.sample.sample_id == 3 and r.sample.version == 4


def test_many_rollouts_all_come_back_once(new_pool):
    pool = new_pool(FAST)
    specs = {pool.submit(Spec(i, i % len(TASKS)), 0, flat()): i for i in range(40)}
    done = drain(pool, list(specs))
    assert sorted(specs[h] for h in done) == list(range(40))
    assert sorted(pool.result(h).sample.sample_id for h in done) == list(range(40))


def test_a_crash_is_raised_by_result_not_by_wait(new_pool):
    pool = new_pool(replace(FAST, crash_prob=1.0))
    h = pool.submit(Spec(0, 0), 0, flat())
    assert drain(pool, [h]) == [h]
    with pytest.raises(WorkerCrashed):
        pool.result(h)


def test_wait_times_out_with_nothing_when_nothing_is_done(new_pool):
    pool = new_pool(replace(FAST, mean_latency_s=1.0, latency_sigma=0.0))
    h = pool.submit(Spec(0, 0), 0, flat())
    started = time.monotonic()
    assert pool.wait([h], timeout=0.05) == []
    assert time.monotonic() - started < 0.8
    assert drain(pool, [h]) == [h]


def test_wait_returns_early_when_one_finishes(new_pool):
    pool = new_pool(FAST)
    h = pool.submit(Spec(0, 0), 0, flat())
    started = time.monotonic()
    assert pool.wait([h], timeout=10.0) == [h]
    assert time.monotonic() - started < 5.0


def test_workers_see_the_weights_of_the_moment_not_later_ones(new_pool):
    pool = new_pool(replace(FAST, mean_latency_s=0.1, latency_sigma=0.0))
    logits = flat()
    logits[0, 0] = 50.0
    h = pool.submit(Spec(0, 0), 0, logits)
    logits[0, 0] = -50.0  # the learner moves on while the worker is still busy
    logits[0, 1] = 50.0
    drain(pool, [h])
    assert pool.result(h).sample.choice == 0


def test_a_new_policy_version_reaches_the_workers(new_pool):
    pool = new_pool(FAST)
    old, new = flat(), flat()
    old[0, 0] = 50.0
    new[0, 3] = 50.0
    a = pool.submit(Spec(0, 0), 0, old)
    b = pool.submit(Spec(1, 0), 1, new)
    drain(pool, [a, b])
    assert (pool.result(a).sample.choice, pool.result(b).sample.choice) == (0, 3)
    assert (pool.result(a).sample.version, pool.result(b).sample.version) == (0, 1)


def test_the_pool_runs_workers_in_parallel(new_pool):
    cfg = replace(FAST, workers=4, mean_latency_s=0.3, latency_sigma=0.0)
    pool = new_pool(cfg)
    started = time.monotonic()
    handles = [pool.submit(Spec(i, 0), 0, flat()) for i in range(4)]
    drain(pool, handles)
    assert time.monotonic() - started < 0.9  # serial would be 1.2


def test_more_rollouts_than_workers_all_run(new_pool):
    cfg = replace(FAST, workers=2, mean_latency_s=0.05, latency_sigma=0.0)
    pool = new_pool(cfg)
    handles = [pool.submit(Spec(i, 0), 0, flat()) for i in range(7)]
    assert len(drain(pool, handles)) == 7


def test_close_can_be_called_twice(new_pool):
    pool = new_pool(FAST)
    pool.close()
    pool.close()


# The factory.


def test_the_factory_builds_a_thread_pool():
    pool = make_pool("thread", FAST)
    try:
        assert isinstance(pool, ThreadPool)
    finally:
        pool.close()


def test_the_factory_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend 'gpu'"):
        make_pool("gpu", FAST)


# The runtime that starts Ray once for several runs.


def test_the_thread_runtime_needs_nothing():
    with runtime("thread"):
        pass


@pytest.mark.ray
def test_the_ray_runtime_leaves_a_ray_that_was_already_running(ray_session):
    with runtime("ray"):
        assert ray_session.is_initialized()
    assert ray_session.is_initialized()


@pytest.mark.ray
def test_the_ray_runtime_stops_the_ray_it_started():
    pytest.importorskip("ray")
    code = (
        "import ray\n"
        "from rollout_fabric.pool import runtime\n"
        "with runtime('ray'):\n"
        "    inside = ray.is_initialized()\n"
        "print(inside, ray.is_initialized())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "True False", out.stderr[-500:]
