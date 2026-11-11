import time
from dataclasses import replace

import numpy as np
import pytest

from rollout_fabric.config import Config
from rollout_fabric.pool import ThreadPool
from rollout_fabric.tasks import N_CHOICES, TASKS
from rollout_fabric.worker import Spec, WorkerCrashed

FAST = Config(workers=2, mean_latency_s=0.0, sandbox=False)


def flat() -> np.ndarray:
    return np.zeros((len(TASKS), N_CHOICES))


def drain(pool, handles, timeout=5.0):
    """Wait until every handle has finished."""
    pending, done = set(handles), []
    deadline = time.monotonic() + timeout
    while pending:
        assert time.monotonic() < deadline, "handles did not finish"
        for h in pool.wait(list(pending), timeout=0.1):
            pending.discard(h)
            done.append(h)
    return done


def test_a_submitted_rollout_comes_back():
    pool = ThreadPool(FAST)
    try:
        h = pool.submit(Spec(3, 1), 4, flat())
        assert drain(pool, [h]) == [h]
        r = pool.result(h)
        assert r.sample.sample_id == 3 and r.sample.version == 4
    finally:
        pool.close()


def test_many_rollouts_all_come_back_once():
    pool = ThreadPool(FAST)
    try:
        specs = {pool.submit(Spec(i, i % len(TASKS)), 0, flat()): i for i in range(40)}
        done = drain(pool, list(specs))
        assert sorted(specs[h] for h in done) == list(range(40))
        assert sorted(pool.result(h).sample.sample_id for h in done) == list(range(40))
    finally:
        pool.close()


def test_a_crash_is_raised_by_result_not_by_wait():
    pool = ThreadPool(replace(FAST, crash_prob=1.0))
    try:
        h = pool.submit(Spec(0, 0), 0, flat())
        assert drain(pool, [h]) == [h]
        with pytest.raises(WorkerCrashed):
            pool.result(h)
    finally:
        pool.close()


def test_wait_times_out_with_nothing_when_nothing_is_done():
    cfg = replace(FAST, mean_latency_s=0.5, latency_sigma=0.0)
    pool = ThreadPool(cfg)
    try:
        h = pool.submit(Spec(0, 0), 0, flat())
        started = time.monotonic()
        assert pool.wait([h], timeout=0.05) == []
        assert time.monotonic() - started < 0.4
        assert drain(pool, [h]) == [h]
    finally:
        pool.close()


def test_wait_returns_early_when_one_finishes():
    pool = ThreadPool(FAST)
    try:
        h = pool.submit(Spec(0, 0), 0, flat())
        started = time.monotonic()
        assert pool.wait([h], timeout=5.0) == [h]
        assert time.monotonic() - started < 1.0
    finally:
        pool.close()


def test_workers_see_the_weights_of_the_moment_not_later_ones():
    pool = ThreadPool(replace(FAST, mean_latency_s=0.1, latency_sigma=0.0))
    try:
        logits = flat()
        logits[0, 0] = 50.0
        h = pool.submit(Spec(0, 0), 0, logits)
        logits[0, 0] = -50.0  # the learner moves on while the worker is still busy
        logits[0, 1] = 50.0
        drain(pool, [h])
        assert pool.result(h).sample.choice == 0
    finally:
        pool.close()


def test_the_pool_runs_workers_in_parallel():
    cfg = replace(FAST, workers=4, mean_latency_s=0.2, latency_sigma=0.0)
    pool = ThreadPool(cfg)
    try:
        started = time.monotonic()
        handles = [pool.submit(Spec(i, 0), 0, flat()) for i in range(4)]
        drain(pool, handles)
        assert time.monotonic() - started < 0.6  # serial would be 0.8
    finally:
        pool.close()


def test_close_can_be_called_twice():
    pool = ThreadPool(FAST)
    pool.close()
    pool.close()
