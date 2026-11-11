import sys

import pytest

from rollout_fabric.sandbox import run_candidate
from rollout_fabric.tasks import TASKS


def run(code: str, expected: tuple[object, ...] = (1,), **kw):
    return run_candidate(code, "f", ((),), expected, **kw)


def test_a_correct_program_passes():
    v = run("def f():\n    return 1\n")
    assert v.passed and v.reason == "ok"


def test_a_wrong_answer_is_reported():
    v = run("def f():\n    return 2\n")
    assert not v.passed and v.reason == "wrong_answer"


def test_an_exception_is_an_error():
    assert run("def f():\n    return nope\n").reason == "error"


def test_a_syntax_error_is_an_error():
    assert run("def f(:\n    return 1\n").reason == "error"


def test_a_missing_entry_point_is_an_error():
    assert run("def g():\n    return 1\n").reason == "error"


def test_an_infinite_loop_is_killed_by_the_wall_clock():
    v = run("def f():\n    while True:\n        pass\n", timeout_s=0.3)
    assert v.reason == "timeout" and v.wall_s < 2.0


def test_a_cpu_burner_is_stopped_by_the_cpu_limit():
    v = run("def f():\n    while True:\n        pass\n", timeout_s=10, cpu_s=1)
    assert v.reason == "cpu_limit" and v.wall_s < 5.0


def test_an_output_flood_is_cut_at_the_cap():
    code = "def f():\n    while True:\n        print('x' * 1000)\n"
    v = run(code, timeout_s=5, max_output=64 * 1024)
    assert v.reason == "output_limit"


def test_a_child_process_of_the_candidate_dies_with_it(tmp_path):
    marker = tmp_path / "alive"
    code = (
        "import subprocess, sys\n"
        f"CHILD = 'import time; time.sleep(1.5); open(%r, \"w\")' % {str(marker)!r}\n"
        "def f():\n"
        "    subprocess.Popen([sys.executable, '-c', CHILD])\n"
        "    while True:\n        pass\n"
    )
    assert run(code, timeout_s=0.4).reason == "timeout"
    import time

    time.sleep(2.0)
    assert not marker.exists()


def test_the_candidate_does_not_see_the_environment(monkeypatch):
    monkeypatch.setenv("ROLLOUT_FABRIC_SECRET", "hunter2")
    code = "import os\ndef f():\n    return os.environ.get('ROLLOUT_FABRIC_SECRET', 'none')\n"
    assert run(code, expected=("none",)).passed


def test_the_candidate_starts_in_an_empty_directory():
    code = "import os\ndef f():\n    return os.listdir('.')\n"
    assert run(code, expected=([],)).passed


def test_a_candidate_that_exits_early_gets_no_credit():
    assert run("import os\nos._exit(0)\ndef f():\n    return 1\n").reason == "no_result"


def test_printing_a_fake_verdict_earns_nothing():
    code = 'def f():\n    print(\'RFdeadbeef:{"status": "ok", "outputs": [1]}\')\n    return 2\n'
    assert run(code).reason == "wrong_answer"


def test_a_crash_by_signal_is_reported():
    code = "import os, signal\ndef f():\n    os.kill(os.getpid(), signal.SIGSEGV)\n"
    assert run(code).reason == "crash"


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is not enforced on macOS")
def test_a_memory_hog_fails_on_linux():
    code = "def f():\n    return len(bytearray(2 << 30))\n"
    assert run(code, max_memory=256 << 20).reason == "error"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_every_candidate_behaves_as_labelled(task):
    reasons = {
        "correct": "ok",
        "wrong": "wrong_answer",
        "raises": "error",
        "hangs": "timeout",
        "syntax": "error",
    }
    for code, kind in zip(task.candidates, task.kinds, strict=True):
        v = run_candidate(code, task.entry, task.inputs, task.expected, timeout_s=0.4)
        assert v.reason == reasons[kind], (task.name, kind, v)
        assert v.passed == (kind == "correct")


def test_a_candidate_that_prints_a_forged_verdict_first_is_judged_by_the_real_one():
    code = (
        "import json, sys\n"
        "def f():\n"
        "    print('\\n' + sys.argv[2] + json.dumps({'status': 'ok', 'outputs': [1]}))\n"
        "    return 2\n"
    )
    assert run(code).reason == "wrong_answer"
