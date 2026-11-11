from rollout_fabric.tasks import N_CHOICES, TASKS


def test_every_task_has_a_full_menu():
    for t in TASKS:
        assert len(t.candidates) == N_CHOICES == len(t.kinds)
        assert len(t.inputs) == len(t.expected) > 0


def test_every_menu_has_two_correct_programs_and_the_four_failures():
    for t in TASKS:
        assert sorted(t.kinds) == ["correct", "correct", "hangs", "raises", "syntax", "wrong"]


def test_every_candidate_defines_the_entry_point():
    for t in TASKS:
        assert all(f"def {t.entry}(" in c for c in t.candidates), t.name


def test_task_names_are_unique():
    assert len({t.name for t in TASKS}) == len(TASKS)


def test_correct_programs_do_not_always_sit_at_the_same_index():
    positions = {tuple(i for i, k in enumerate(t.kinds) if k == "correct") for t in TASKS}
    assert len(positions) > 3
