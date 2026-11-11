import pytest

from rollout_fabric.fabric import RunStats
from rollout_fabric.report import comparison, markdown_table, median_summary, summarize


def stats(**kw) -> RunStats:
    base = dict(
        wall_s=10.0,
        learner_busy_s=2.0,
        consumed=100,
        dropped_stale=4,
        crashes=1,
        staleness=[0, 1, 1, 1],
        step_rewards=[0.5, 0.9, 1.0, 1.0],
        step_end_s=[2, 4, 6, 8],
    )
    return RunStats("x", **{**base, **kw})


def test_a_summary_holds_the_numbers_of_a_run():
    s = summarize(stats(), target=0.9)
    assert (s.wall_s, s.samples_per_s, s.learner_idle) == (10.0, 10.0, 0.8)
    assert (s.dropped_stale, s.crashes) == (4, 1)
    assert s.mean_staleness == 0.75
    assert s.to_target_s == 8
    assert s.reached == 1 and s.runs == 1


def test_a_run_that_never_reaches_the_target_has_no_time():
    s = summarize(stats(step_rewards=[0.1, 0.2]), target=0.9)
    assert s.to_target_s is None and s.reached == 0


def test_the_median_is_taken_column_by_column():
    runs = [summarize(stats(wall_s=w, consumed=100), 0.9) for w in (10.0, 20.0, 40.0)]
    m = median_summary(runs)
    assert m.wall_s == 20.0
    assert m.samples_per_s == 5.0
    assert m.runs == 3 and m.reached == 3


def test_the_median_time_to_target_counts_only_the_runs_that_got_there():
    good = summarize(stats(step_end_s=[1, 2, 3, 4]), 0.9)
    slow = summarize(stats(step_end_s=[2, 4, 6, 12]), 0.9)
    bad = summarize(stats(step_rewards=[0.1, 0.1, 0.1, 0.1]), 0.9)
    m = median_summary([good, slow, bad])
    assert m.to_target_s == 8.0 and m.reached == 2 and m.runs == 3


def test_the_median_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        median_summary([])


def test_the_table_has_a_header_a_rule_and_one_row_per_mode():
    rows = {"sync": summarize(stats(), 0.9), "async": summarize(stats(wall_s=5.0), 0.9)}
    lines = markdown_table(rows, 0.9).splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("| mode |") and "seconds to reward 0.9" in lines[0]
    assert lines[1].startswith("|:--|")
    assert lines[2].startswith("| sync | 10.0 | 10.0 | 80% | 0.75 / 1 | 4 | 1 | 8.0 | 0.85 |")
    assert lines[3].startswith("| async | 5.0 | 20.0 |")
    assert {line.count("|") for line in lines} == {10}


def test_the_table_says_when_a_target_was_not_reached_and_by_how_many_runs():
    never = summarize(stats(step_rewards=[0.1, 0.1]), 0.9)
    assert "not reached" in markdown_table({"sync": never}, 0.9)
    some = median_summary([summarize(stats(), 0.9), never, summarize(stats(), 0.9)])
    assert "(2/3 runs)" in markdown_table({"sync": some}, 0.9)


def test_the_comparison_gives_the_ratios():
    sync = summarize(stats(wall_s=10.0, step_end_s=[2, 4, 6, 8]), 0.9)
    asyn = summarize(stats(wall_s=4.0, learner_busy_s=2.0, step_end_s=[1, 2, 3, 4]), 0.9)
    text = comparison(sync, asyn, 0.9)
    assert "2.50x samples/s" in text
    assert "learner idle 80% -> 50%" in text
    assert "2.00x faster to reward 0.9 (8.0 s -> 4.0 s)" in text


def test_the_comparison_leaves_out_a_ratio_it_cannot_compute():
    never = summarize(stats(step_rewards=[0.1]), 0.9)
    assert "faster to reward" not in comparison(never, summarize(stats(), 0.9), 0.9)
