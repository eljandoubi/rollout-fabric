import json
import socket

from rollout_fabric.config import Config
from rollout_fabric.fabric import RunStats, run_sync
from rollout_fabric.metrics import (
    GAUGES,
    dogstatsd_line,
    dogstatsd_sink,
    fanout,
    jsonl_sink,
    prometheus_text,
)
from rollout_fabric.pool import ThreadPool

EVENT = {
    "mode": "async",
    "step": 3,
    "reward": 0.5,
    "consumed": 48,
    "dropped_stale": 2,
    "crashes": 1,
    "mean_staleness": 0.75,
    "elapsed_s": 1.25,
    "learner_busy_s": 0.5,
}


def test_the_jsonl_sink_appends_one_object_per_step(tmp_path):
    path = tmp_path / "steps.jsonl"
    emit = jsonl_sink(path)
    emit(EVENT)
    emit({**EVENT, "step": 4})
    lines = path.read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [3, 4]
    assert json.loads(lines[0]) == EVENT


def test_the_jsonl_sink_keeps_earlier_lines(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"old": true}\n')
    jsonl_sink(path)(EVENT)
    assert path.read_text().splitlines()[0] == '{"old": true}'


def test_the_fanout_feeds_every_sink_in_order():
    seen = []
    emit = fanout(
        [lambda e: seen.append(("a", e["step"])), lambda e: seen.append(("b", e["step"]))]
    )
    emit(EVENT)
    assert seen == [("a", 3), ("b", 3)]


def test_no_sinks_means_no_callback():
    assert fanout([]) is None


def test_a_dogstatsd_gauge_line_has_name_value_type_and_tags():
    assert dogstatsd_line("rf", "reward", 0.5, ["mode:sync"]) == "rf.reward:0.5|g|#mode:sync"
    assert dogstatsd_line("rf", "reward", 0.5, []) == "rf.reward:0.5|g"
    assert dogstatsd_line("rf", "crashes", 3.0, ["a:b", "c:d"]) == "rf.crashes:3|g|#a:b,c:d"


def test_the_dogstatsd_sink_sends_the_gauges_over_udp():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2.0)
    try:
        emit = dogstatsd_sink("127.0.0.1", server.getsockname()[1])
        emit(EVENT)
        lines = server.recv(4096).decode().splitlines()
    finally:
        server.close()
    assert "rollout_fabric.reward:0.5|g|#mode:async" in lines
    assert "rollout_fabric.dropped_stale:2|g|#mode:async" in lines
    assert "rollout_fabric.elapsed_s:1.25|g|#mode:async" in lines
    assert all(line.endswith("|#mode:async") for line in lines)


def test_a_dead_metrics_endpoint_never_stops_the_run():
    # nothing listens on this port; UDP does not care, and a failing send must not raise either
    emit = dogstatsd_sink("127.0.0.1", 9)
    emit(EVENT)
    unresolvable = dogstatsd_sink("no-such-host.invalid", 8125)
    unresolvable(EVENT)


def test_the_prometheus_text_has_help_type_and_a_value_per_metric():
    stats = RunStats(
        "async",
        wall_s=4.0,
        learner_busy_s=1.0,
        consumed=100,
        dropped_stale=3,
        crashes=2,
        staleness=[0, 1, 2],
        step_rewards=[0.2, 0.9, 0.95, 0.95],
        step_end_s=[1, 2, 3, 4],
    )
    stats.reasons.update({"ok": 70, "wrong_answer": 30})
    text = prometheus_text(stats, target=0.9)
    assert "# TYPE rollout_fabric_samples_consumed_total counter" in text
    assert 'rollout_fabric_samples_consumed_total{mode="async"} 100' in text
    assert 'rollout_fabric_samples_dropped_stale_total{mode="async"} 3' in text
    assert 'rollout_fabric_worker_crashes_total{mode="async"} 2' in text
    assert 'rollout_fabric_samples_per_second{mode="async"} 25' in text
    assert 'rollout_fabric_learner_idle_ratio{mode="async"} 0.75' in text
    assert 'rollout_fabric_seconds_to_target_reward{mode="async"} 4' in text
    assert 'rollout_fabric_verdicts_total{mode="async",reason="ok"} 70' in text
    assert 'rollout_fabric_verdicts_total{mode="async",reason="wrong_answer"} 30' in text
    assert text.endswith("\n")


def test_every_prometheus_sample_line_is_well_formed():
    text = prometheus_text(RunStats("sync", wall_s=1.0, consumed=5))
    for line in text.splitlines():
        if line.startswith("#"):
            assert line.startswith(("# HELP ", "# TYPE "))
        else:
            name_and_labels, value = line.rsplit(" ", 1)
            assert name_and_labels.startswith("rollout_fabric_") and "{" in name_and_labels
            float(value)


def test_the_target_time_is_left_out_when_the_target_was_never_reached():
    text = prometheus_text(RunStats("sync", step_rewards=[0.1], step_end_s=[1.0]))
    assert "seconds_to_target_reward" not in text


def test_the_controller_emits_every_gauge_the_sink_reads():
    cfg = Config(
        workers=2, batch_size=4, steps=2, mean_latency_s=0.0, train_step_s=0.0, sandbox=False
    )
    seen = []
    pool = ThreadPool(cfg)
    try:
        run_sync(cfg, pool, on_step=seen.append)
    finally:
        pool.close()
    assert len(seen) == 2
    assert all(set(GAUGES) <= set(event) for event in seen)
