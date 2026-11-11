import numpy as np
import pytest

from rollout_fabric.policy import Policy, Sample, sample_choice, softmax


def sample(task: int, choice: int, reward: float, behavior_logp: float, version: int = 0) -> Sample:
    return Sample(0, task, choice, reward, behavior_logp, version)


def test_softmax_sums_to_one_and_ignores_a_shift():
    z = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.allclose(p, softmax(z + 1000.0))


def test_sampling_follows_the_probabilities():
    logits = np.array([[0.0, np.log(3.0)]])
    rng = np.random.default_rng(0)
    draws = [sample_choice(logits, 0, rng)[0] for _ in range(20000)]
    assert abs(np.mean(draws) - 0.75) < 0.02


def test_sampling_returns_the_log_probability_of_the_choice():
    logits = np.array([[0.0, 1.0, 2.0]])
    choice, logp = sample_choice(logits, 0, np.random.default_rng(3))
    assert logp == pytest.approx(np.log(softmax(logits[0])[choice]))


def test_update_makes_the_rewarded_choice_more_likely_and_counts_versions():
    pol = Policy(n_tasks=2, n_choices=3)
    logp = float(np.log(1 / 3))
    batch = [sample(0, 2, 1.0, logp), sample(0, 0, 0.0, logp), sample(0, 1, 0.0, logp)]
    before = pol.probs(0)[2]
    pol.update(batch)
    assert pol.probs(0)[2] > before
    assert pol.version == 1


def test_update_leaves_other_tasks_alone():
    pol = Policy(n_tasks=2, n_choices=3)
    pol.update([sample(0, 1, 1.0, float(np.log(1 / 3))), sample(0, 0, 0.0, float(np.log(1 / 3)))])
    assert np.all(pol.logits[1] == 0.0)


def test_equal_rewards_change_nothing():
    pol = Policy(n_tasks=1, n_choices=3)
    pol.update([sample(0, 0, 1.0, float(np.log(1 / 3))), sample(0, 1, 1.0, float(np.log(1 / 3)))])
    assert np.all(pol.logits == 0.0)


def test_empty_batch_is_refused():
    with pytest.raises(ValueError):
        Policy(1, 2).update([])


def test_update_follows_the_gradient_of_the_expected_reward():
    # With every choice once in the batch and uniform logits, the step is lr * grad J(theta).
    rewards = np.array([1.0, 0.0, 0.5, 0.0])
    pol = Policy(n_tasks=1, n_choices=4, lr=1.0)
    logp = float(np.log(1 / 4))
    pol.update([sample(0, k, float(r), logp) for k, r in enumerate(rewards)])

    def expected_reward(theta: np.ndarray) -> float:
        return float(softmax(theta) @ rewards)

    eps = 1e-6
    numeric = np.array(
        [
            (expected_reward(np.eye(4)[j] * eps) - expected_reward(-np.eye(4)[j] * eps)) / (2 * eps)
            for j in range(4)
        ]
    )
    assert np.allclose(pol.logits[0], numeric, atol=1e-6)


def test_a_stale_sample_is_weighted_by_the_clipped_ratio():
    pol = Policy(n_tasks=1, n_choices=2, clip=2.0)
    stale = sample(0, 0, 1.0, behavior_logp=float(np.log(1e-6)))  # now far likelier than then
    other = sample(0, 1, 0.0, behavior_logp=float(np.log(0.5)))
    mean_ratio = pol.update([stale, other])
    assert mean_ratio == pytest.approx((2.0 + 1.0) / 2)


def test_a_fresh_sample_has_ratio_one():
    pol = Policy(n_tasks=1, n_choices=2)
    assert (
        pol.update([sample(0, 0, 1.0, float(np.log(0.5))), sample(0, 1, 0.0, float(np.log(0.5)))])
        == 1.0
    )
