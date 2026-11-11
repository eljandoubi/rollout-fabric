"""Where rollouts run. Two pools share one small interface: threads, and Ray actors."""

from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Collection
from typing import Any, Protocol

import numpy as np

from rollout_fabric.config import Config
from rollout_fabric.policy import Logits
from rollout_fabric.worker import Rollout, Spec, generate

Handle = Any  # opaque token for one submitted rollout


class Pool(Protocol):
    def submit(self, spec: Spec, version: int, logits: Logits) -> Handle: ...

    def wait(self, handles: Collection[Handle], timeout: float) -> list[Handle]:
        """The handles that have finished, waiting up to `timeout` s for the first one."""
        ...

    def result(self, handle: Handle) -> Rollout:
        """The rollout of a finished handle. Raises WorkerCrashed if the worker died."""
        ...

    def close(self) -> None: ...


class ThreadPool:
    """Runs workers as threads of this process. Fast to start, so the logic tests use it."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pool = cf.ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="rollout")
        self._version = -1
        self._logits: Logits = np.zeros(0)

    def submit(self, spec: Spec, version: int, logits: Logits) -> Handle:
        if version != self._version:  # one snapshot per policy version, like a broadcast
            self._version, self._logits = version, logits.copy()
        return self._pool.submit(generate, spec, version, self._logits, self.cfg)

    def wait(self, handles: Collection[Handle], timeout: float) -> list[Handle]:
        done, _ = cf.wait(handles, timeout=timeout, return_when=cf.FIRST_COMPLETED)
        return list(done)

    def result(self, handle: Handle) -> Rollout:
        rollout: Rollout = handle.result()
        return rollout

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
