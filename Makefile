PY ?= python

.PHONY: install fmt lint type test test-ray cov check mutate bench docker clean

install:
	$(PY) -m pip install -e ".[dev]"

fmt:
	ruff format .
	ruff check --fix .

lint:
	ruff format --check .
	ruff check .

type:
	mypy

test:
	pytest -q -m "not ray"

test-ray:
	pytest -q -m ray

cov:
	pytest -q -m "not ray" --cov --cov-report=term-missing --cov-fail-under=90

check: lint type cov

mutate:
	$(PY) scripts/mutate.py

bench:
	rollout-fabric compare --repeat 3 --no-sandbox --workers 8

docker:
	docker build -t rollout-fabric .
	docker run --rm rollout-fabric compare --quick

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
