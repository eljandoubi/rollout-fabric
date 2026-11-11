"""Run candidate code in a throw-away subprocess with resource limits.

This is a guard rail for generated code, not a security boundary: the child shares the kernel, the
network and the user's files with the host. A real deployment puts the same interface in front of a
container, gVisor or Firecracker.

The expected answers stay in the parent. The child only gets the inputs and returns what the
candidate computed, behind a token it printed last, so printing a fake verdict earns nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CHILD = r"""
import json, resource, sys
cpu, fsize, nofile, mem = json.loads(sys.argv[1])
token = sys.argv[2]
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
if mem and sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
req = json.loads(sys.stdin.read())
try:
    ns = {"__name__": "candidate"}
    exec(compile(req["code"], "<candidate>", "exec"), ns)
    fn = ns[req["entry"]]
    outputs = [fn(*args) for args in req["inputs"]]
    verdict = {"status": "ok", "outputs": json.loads(json.dumps(outputs))}
except BaseException as exc:
    verdict = {"status": "error", "detail": type(exc).__name__}
sys.stdout.write("\n" + token + json.dumps(verdict) + "\n")
sys.stdout.flush()
"""

_ENV = {"PATH": "/usr/bin:/bin"}


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str  # ok, wrong_answer, error, timeout, cpu_limit, output_limit, crash, no_result
    wall_s: float


def run_candidate(
    code: str,
    entry: str,
    inputs: tuple[tuple[Any, ...], ...],
    expected: tuple[Any, ...],
    *,
    timeout_s: float = 1.0,
    cpu_s: int = 2,
    max_output: int = 1 << 20,
    max_memory: int = 512 << 20,
) -> Verdict:
    """Call `entry` of `code` on every input in a fresh process and compare with `expected`."""
    token = "RF" + secrets.token_hex(8) + ":"
    request = json.dumps({"code": code, "entry": entry, "inputs": inputs}).encode()
    limits = json.dumps([cpu_s, max_output, 64, max_memory])
    with tempfile.TemporaryDirectory(prefix="rf-sandbox-") as tmp:
        work = Path(tmp, "work")
        work.mkdir()
        out_file = Path(tmp, "stdout")
        start = time.monotonic()
        with out_file.open("wb") as out:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _CHILD, limits, token],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=subprocess.DEVNULL,
                cwd=work,
                env=_ENV,
                start_new_session=True,
            )
            try:
                proc.communicate(request, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                return Verdict(False, "timeout", time.monotonic() - start)
        wall = time.monotonic() - start
        return _judge(proc.returncode, out_file, token, expected, wall, max_output)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    # the candidate may have started children of its own
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.communicate()


def _judge(
    returncode: int, out_file: Path, token: str, expected: tuple[Any, ...], wall: float, cap: int
) -> Verdict:
    if returncode < 0:
        reason = "cpu_limit" if -returncode == signal.SIGXCPU else "crash"
        return Verdict(False, reason, wall)
    if out_file.stat().st_size >= cap:
        return Verdict(False, "output_limit", wall)
    text = out_file.read_bytes().decode(errors="replace")
    at = text.rfind(token)
    if at < 0:
        return Verdict(False, "no_result", wall)
    try:
        verdict = json.loads(text[at + len(token) :].splitlines()[0])
    except (IndexError, ValueError):
        return Verdict(False, "no_result", wall)
    if verdict.get("status") != "ok":
        return Verdict(False, "error", wall)
    ok = bool(verdict["outputs"] == json.loads(json.dumps(expected)))
    return Verdict(ok, "ok" if ok else "wrong_answer", wall)
