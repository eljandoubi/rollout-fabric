"""Mutation check: change one line of the code at a time and see whether the tests notice.

A surviving mutant is a behaviour that no test pins down. Run it from the repository root:

    python scripts/mutate.py            # the fast mutants (no Ray)
    python scripts/mutate.py --ray      # also the Ray mutants (slow)
    python scripts/mutate.py F01 W02    # only these ids

The mutants are applied in a copy of the repository, never in the working tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (id, file, original line fragment, replacement, what the mutant breaks)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    (
        "F01",
        "fabric.py",
        "run.policy.version - r.sample.version <= cfg.max_staleness]",
        "run.policy.version - r.sample.version <= cfg.max_staleness + 1]",
        "staleness bound off by one",
    ),
    (
        "F02",
        "fabric.py",
        "run.stats.dropped_stale += len(buffer) - len(fresh)",
        "run.stats.dropped_stale += 0",
        "dropped samples not counted",
    ),
    (
        "F03",
        "fabric.py",
        "cap = (cfg.max_staleness + 1) * cfg.batch_size",
        "cap = 10**9",
        "no limit on how far dispatch runs ahead",
    ),
    (
        "F04",
        "fabric.py",
        "max_inflight = cfg.workers * cfg.inflight_per_worker",
        "max_inflight = cfg.workers * cfg.inflight_per_worker * 4",
        "queue deeper than configured",
    ),
    (
        "F05",
        "fabric.py",
        "buffer.sort(key=lambda r: (r.sample.version, r.sample.sample_id))",
        "buffer.sort(key=lambda r: (-r.sample.version, r.sample.sample_id))",
        "newest samples first",
    ),
    (
        "F06",
        "fabric.py",
        "again = replace(spec, attempt=spec.attempt + 1)",
        "again = replace(spec, attempt=spec.attempt)",
        "a retry repeats the same draws",
    ),
    ("F07", "fabric.py", "self.stats.crashes += 1", "pass", "crashes not counted"),
    (
        "F08",
        "fabric.py",
        "self.stats.staleness.append(self.policy.version - r.sample.version)",
        "self.stats.staleness.append(0)",
        "staleness always reported as zero",
    ),
    (
        "F09",
        "fabric.py",
        "while run.policy.version < cfg.steps:",
        "while run.policy.version <= cfg.steps:",
        "one step too many",
    ),
    (
        "F10",
        "fabric.py",
        "for _ in range(cfg.batch_size):",
        "for _ in range(cfg.batch_size - 1):",
        "sync batch one sample short",
    ),
    (
        "F11",
        "fabric.py",
        "stats.learner_busy_s += now - started",
        "stats.learner_busy_s += 0.0",
        "learner never busy",
    ),
    (
        "F12",
        "fabric.py",
        "self.step_rewards[i - window + 1 : i + 1]) >= target",
        "self.step_rewards[i - window + 1 : i + 1]) > target",
        "target must be beaten, not reached",
    ),
    (
        "F13",
        "fabric.py",
        "np.percentile(self.staleness, 95)",
        "np.percentile(self.staleness, 50)",
        "p95 is the median",
    ),
    (
        "F14",
        "fabric.py",
        "if len(buffer) >= cfg.batch_size:",
        "if len(buffer) > cfg.batch_size:",
        "async waits for one extra sample",
    ),
    (
        "F15",
        "fabric.py",
        "if self._crashes[spec.sample_id] >= self.cfg.max_attempts:",
        "if self._crashes[spec.sample_id] > self.cfg.max_attempts:",
        "one retry too many",
    ),
    (
        "W01",
        "worker.py",
        "np.random.default_rng([cfg.seed, spec.sample_id, spec.attempt])",
        "np.random.default_rng([cfg.seed, spec.sample_id])",
        "attempt does not change the draws",
    ),
    (
        "W02",
        "worker.py",
        "reward = 1.0 if verdict.passed else 0.0",
        "reward = 1.0",
        "every candidate is rewarded",
    ),
    (
        "W03",
        "worker.py",
        "mu = np.log(cfg.mean_latency_s) - cfg.latency_sigma**2 / 2",
        "mu = np.log(cfg.mean_latency_s)",
        "latency mean is wrong",
    ),
    (
        "W04",
        "worker.py",
        "rng.random() < cfg.crash_prob",
        "rng.random() > cfg.crash_prob",
        "crash probability inverted",
    ),
    (
        "W05",
        "worker.py",
        "Sample(spec.sample_id, spec.task, choice, reward, logp, version)",
        "Sample(spec.sample_id, spec.task, choice, reward, logp, version + 1)",
        "sample claims a newer policy",
    ),
    (
        "P01",
        "pool.py",
        "self._version, self._logits = version, logits.copy()",
        "self._version, self._logits = version, logits",
        "workers share the learner's weights",
    ),
    (
        "P02",
        "pool.py",
        "if version != self._version:",
        "if self._version < 0:",
        "weights never refreshed",
    ),
    (
        "Y01",
        "policy.py",
        "min(float(np.exp(np.log(p[s.choice]) - s.behavior_logp)), self.clip)",
        "float(np.exp(np.log(p[s.choice]) - s.behavior_logp))",
        "importance ratio not clipped",
    ),
    (
        "Y02",
        "policy.py",
        "baseline = float(np.mean([s.reward for s in batch]))",
        "baseline = 0.0",
        "no baseline",
    ),
    ("Y03", "policy.py", "self.version += 1", "self.version += 2", "version jumps"),
    (
        "S01",
        "sandbox.py",
        "resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))",
        "pass",
        "no CPU limit",
    ),
    (
        "S02",
        "sandbox.py",
        "start_new_session=True,",
        "start_new_session=False,",
        "candidate shares our process group",
    ),
    ("S03", "sandbox.py", "env=_ENV,", "env=None,", "candidate sees our environment"),
    (
        "S04",
        "sandbox.py",
        "at = text.rfind(token)",
        "at = text.find(token)",
        "first marker wins, so a candidate can forge a result",
    ),
]

RAY_MUTANTS: list[tuple[str, str, str, str, str]] = [
    (
        "R01",
        "ray_pool.py",
        "os.kill(os.getpid(), signal.SIGKILL)",
        "raise",
        "an injected crash is only an exception",
    ),
    ("R02", "ray_pool.py", "self._idle.append(job.actor)", "pass", "workers never become idle"),
    (
        "R03",
        "ray_pool.py",
        "if version != self._version:",
        "if self._version < 0:",
        "weights never refreshed",
    ),
]


def run_tests(workdir: Path, ray: bool, timeout: float) -> tuple[str, str]:
    """('killed' | 'survived' | 'hang', first failing test or '')."""
    marker = "ray" if ray else "not ray"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        "-m",
        marker,
        "-p",
        "no:cacheprovider",
        "--no-header",
        "-rf",
    ]
    env = {**os.environ, "PYTHONPATH": str(workdir / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        out = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "hang", ""
    if out.returncode == 0:
        return "survived", ""
    failed = [ln for ln in out.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return "killed", (failed[0][:100] if failed else "collection error")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="run only these mutant ids")
    parser.add_argument("--ray", action="store_true", help="include the Ray mutants")
    parser.add_argument("--timeout", type=float, default=120.0, help="per mutant, in seconds")
    args = parser.parse_args()

    mutants = MUTANTS + (
        RAY_MUTANTS if args.ray or any(i.startswith("R") for i in args.ids) else []
    )
    if args.ids:
        mutants = [m for m in mutants if m[0] in args.ids]
    bad = 0
    with tempfile.TemporaryDirectory(prefix="rf-mutants-") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(
            ROOT,
            work,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".mypy_cache",
                ".ruff_cache",
                ".pytest_cache",
                "*.egg-info",
                ".coverage",
            ),
        )
        baseline, why = run_tests(work, ray=False, timeout=args.timeout)
        if baseline != "survived":
            print(f"the unmutated tests do not pass ({baseline}: {why}); fix that first")
            return 2
        for mid, name, old, new, what in mutants:
            path = work / "src" / "rollout_fabric" / name
            original = path.read_text()
            if original.count(old) != 1:
                print(f"{mid}: anchor found {original.count(old)} times in {name}: {old!r}")
                bad += 1
                continue
            path.write_text(original.replace(old, new))
            started = time.monotonic()
            try:
                verdict, why = run_tests(work, ray=mid.startswith("R"), timeout=args.timeout)
            finally:
                path.write_text(original)
            took = time.monotonic() - started
            print(
                f"{mid} {verdict:8s} {took:5.1f}s  {what}" + (f"  [{why}]" if why else ""),
                flush=True,
            )
            bad += verdict == "survived"
    print(f"\n{len(mutants) - bad} of {len(mutants)} mutants killed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
