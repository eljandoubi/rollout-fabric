"""Six small coding tasks, each with a fixed menu of candidate programs the policy chooses from.

The menu of every task holds two correct programs and four ways to fail: a wrong answer, an
exception, an infinite loop and a syntax error. The hanging candidate is what makes some rollouts
slow, the way a runaway program slows a real code-reward pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

N_CHOICES = 6


@dataclass(frozen=True)
class Task:
    name: str
    entry: str
    inputs: tuple[tuple[Any, ...], ...]
    expected: tuple[Any, ...]  # compared by the caller, never sent to the sandbox
    candidates: tuple[str, ...]
    kinds: tuple[str, ...]  # correct, wrong, raises, hangs or syntax, one per candidate


def _task(
    name: str,
    params: str,
    inputs: tuple[tuple[Any, ...], ...],
    expected: tuple[Any, ...],
    good: str,
    other_good: str,
    bad: str,
    rotate: int,
) -> Task:
    def fn(body: str) -> str:
        return f"def {name}({params}):\n    {body}\n"

    candidates = [
        fn(good),
        fn(bad),
        fn("return undefined_name"),
        fn("while True:\n        pass"),
        fn(other_good),
        f"def {name}({params}\n    return 0\n",
    ]
    kinds = ["correct", "wrong", "raises", "hangs", "correct", "syntax"]
    k = rotate % N_CHOICES
    return Task(
        name,
        name,
        inputs,
        expected,
        tuple(candidates[k:] + candidates[:k]),
        tuple(kinds[k:] + kinds[:k]),
    )


TASKS: tuple[Task, ...] = (
    _task(
        "add",
        "a, b",
        ((1, 2), (5, 7), (-3, 3), (0, 0)),
        (3, 12, 0, 0),
        "return a + b",
        "return sum((a, b))",
        "return a - b",
        0,
    ),
    _task(
        "square",
        "x",
        ((3,), (0,), (-4,), (10,)),
        (9, 0, 16, 100),
        "return x * x",
        "return x ** 2",
        "return x + x",
        1,
    ),
    _task(
        "reverse",
        "s",
        (("abc",), ("",), ("xy z",)),
        ("cba", "", "z yx"),
        "return s[::-1]",
        "return ''.join(reversed(s))",
        "return s",
        2,
    ),
    _task(
        "is_even",
        "n",
        ((2,), (7,), (0,), (-4,)),
        (True, False, True, True),
        "return n % 2 == 0",
        "return not n & 1",
        "return n % 2 == 1",
        3,
    ),
    _task(
        "biggest",
        "xs",
        (([1, 5, 2],), ([-3, -1],), ([7],)),
        (5, -1, 7),
        "return max(xs)",
        "return sorted(xs)[-1]",
        "return xs[0]",
        4,
    ),
    _task(
        "fib",
        "n",
        ((0,), (1,), (10,), (15,)),
        (0, 1, 55, 610),
        "return n if n < 2 else fib(n - 1) + fib(n - 2)",
        "a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "return n * 2",
        5,
    ),
)
