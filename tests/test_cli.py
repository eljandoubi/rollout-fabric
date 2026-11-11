import json
import socket

import pytest

from rollout_fabric.cli import build_parser, main

SMALL = [
    "--workers",
    "2",
    "--batch-size",
    "4",
    "--steps",
    "4",
    "--mean-latency-s",
    "0.0",
    "--train-step-s",
    "0.0",
    "--no-sandbox",
]


def test_run_prints_a_one_row_table(capsys):
    assert main(["run", "--mode", "sync", *SMALL]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[2].startswith("| sync |")


def test_run_defaults_to_async(capsys):
    main(["run", *SMALL])
    assert capsys.readouterr().out.splitlines()[2].startswith("| async |")


def test_run_writes_one_jsonl_line_per_step(tmp_path, capsys):
    path = tmp_path / "steps.jsonl"
    main(["run", "--mode", "sync", "--jsonl", str(path), *SMALL])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["mode"] == "sync" for r in rows)


def test_run_writes_the_prometheus_file(tmp_path, capsys):
    path = tmp_path / "rf.prom"
    main(["run", "--mode", "sync", "--prom-file", str(path), *SMALL])
    text = path.read_text()
    assert 'rollout_fabric_samples_consumed_total{mode="sync"} 16' in text


def test_run_sends_dogstatsd_packets_one_per_step(capsys):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2.0)
    try:
        main(["run", "--mode", "sync", "--statsd", f"127.0.0.1:{server.getsockname()[1]}", *SMALL])
        packets = [server.recv(4096).decode() for _ in range(4)]
    finally:
        server.close()
    assert all("rollout_fabric.reward:" in p and "|#mode:sync" in p for p in packets)


@pytest.mark.parametrize("address", ["localhost", "host:", ":8125", "host:port"])
def test_a_bad_statsd_address_is_refused(address):
    with pytest.raises(SystemExit):
        main(["run", "--statsd", address, *SMALL])


def test_compare_prints_both_modes_and_the_ratio(capsys):
    assert main(["compare", "--quick", *SMALL[:4], "--no-sandbox", "--steps", "6"]) == 0
    out = capsys.readouterr().out
    assert "| sync |" in out and "| async |" in out
    assert "async vs sync:" in out and "x samples/s" in out
    assert "median of" not in out


def test_compare_repeat_reports_the_median(capsys):
    main(
        [
            "compare",
            "--quick",
            "--repeat",
            "2",
            "--workers",
            "2",
            "--batch-size",
            "4",
            "--steps",
            "4",
            "--no-sandbox",
        ]
    )
    assert "(median of 2 runs per mode)" in capsys.readouterr().out


def test_quick_shortens_the_timings():
    from rollout_fabric.cli import _config

    cfg = _config(build_parser().parse_args(["compare", "--quick"]))
    assert (cfg.steps, cfg.mean_latency_s, cfg.train_step_s) == (12, 0.02, 0.01)
    assert _config(build_parser().parse_args(["compare"])).steps == 40


def test_every_config_field_has_an_option_with_the_same_default():
    args = build_parser().parse_args(["run"])
    from dataclasses import fields

    from rollout_fabric.cli import _config
    from rollout_fabric.config import Config

    assert _config(args) == Config()
    assert {f.name for f in fields(Config)} <= set(vars(args))


def test_a_flag_overrides_the_default():
    from rollout_fabric.cli import _config

    args = build_parser().parse_args(["run", "--max-staleness", "5", "--no-sandbox", "--lr", "1.5"])
    cfg = _config(args)
    assert (cfg.max_staleness, cfg.sandbox, cfg.lr) == (5, False, 1.5)


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.parametrize("bad", [["run", "--mode", "weird"], ["run", "--backend", "gpu"]])
def test_unknown_choices_are_refused(bad):
    with pytest.raises(SystemExit):
        main(bad)
