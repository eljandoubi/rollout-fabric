"""Ray actors as workers. `pip install rollout-fabric[ray]`.

Work is pulled, not pushed: a rollout goes to an actor only when that actor is idle, so a
straggler never holds up work that is queued behind it, and a dead worker loses at most the one
rollout it was generating. An injected crash is a real one: the actor kills its own process, Ray
restarts it (`max_restarts=-1`), and the controller sees a `WorkerCrashed` and asks again.
"""

from __future__ import annotations

import logging
import os
import signal
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from rollout_fabric.config import Config
from rollout_fabric.policy import Logits
from rollout_fabric.worker import Rollout, Spec, WorkerCrashed, generate


@dataclass(eq=False)
class _Job:
    spec: Spec
    version: int
    weights: Any  # ObjectRef of the logits, shared by every job of one policy version
    ref: Any = None  # ObjectRef of the result once an actor has taken the job
    actor: Any = None
    done: bool = False


def _worker_class(ray: Any) -> Any:
    @ray.remote(num_cpus=0, max_restarts=-1)
    class Worker:
        def ping(self) -> int:
            return os.getpid()

        def run(self, spec: Spec, version: int, logits: Logits, cfg: Config) -> Rollout:
            try:
                return generate(spec, version, logits, cfg)
            except WorkerCrashed:
                os.kill(os.getpid(), signal.SIGKILL)  # die like an OOM-killed process would
                raise  # not reached

    return Worker


class RayPool:
    def __init__(self, cfg: Config, num_cpus: int | None = None) -> None:
        import ray

        self._ray = ray
        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            ray.init(
                num_cpus=num_cpus,
                include_dashboard=False,
                log_to_driver=False,
                logging_level=logging.ERROR,
            )
        self.cfg = cfg
        self._cfg_ref = ray.put(cfg)
        worker = _worker_class(ray)
        self._actors = [worker.remote() for _ in range(cfg.workers)]
        ray.get([a.ping.remote() for a in self._actors])  # wait until every worker is up
        self._idle = list(self._actors)
        self._queue: deque[_Job] = deque()
        self._running: dict[Any, _Job] = {}
        self._version = -1
        self._weights: Any = None

    def submit(self, spec: Spec, version: int, logits: Logits) -> _Job:
        if version != self._version:  # one broadcast per policy version
            self._version, self._weights = version, self._ray.put(logits.copy())
        job = _Job(spec, version, self._weights)
        self._queue.append(job)
        self._pump()
        return job

    def _pump(self) -> None:
        while self._queue and self._idle:
            job, actor = self._queue.popleft(), self._idle.pop()
            job.actor = actor
            job.ref = actor.run.remote(job.spec, job.version, job.weights, self._cfg_ref)
            self._running[job.ref] = job

    def wait(self, handles: Collection[_Job], timeout: float) -> list[_Job]:
        if not any(j.done for j in handles) and self._running:
            refs = [j.ref for j in handles if j.ref is not None and not j.done]
            ready, _ = self._ray.wait(refs, num_returns=1, timeout=timeout) if refs else ([], [])
            for ref in ready:
                job = self._running.pop(ref)
                job.done = True
                self._idle.append(job.actor)  # a crashed actor is already being restarted
            self._pump()
        return [j for j in handles if j.done]

    def result(self, handle: _Job) -> Rollout:
        try:
            rollout: Rollout = self._ray.get(handle.ref)
        except self._ray.exceptions.RayActorError as e:
            raise WorkerCrashed(f"sample {handle.spec.sample_id}: worker died") from e
        return rollout

    def close(self) -> None:
        for actor in self._actors:
            self._ray.kill(actor)
        self._actors = []
        if self._owns_ray and self._ray.is_initialized():
            self._ray.shutdown()
