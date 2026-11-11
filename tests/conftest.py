import logging

import pytest


@pytest.fixture(scope="session")
def ray_session():
    """One local Ray for the whole run: starting it costs seconds."""
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, logging_level=logging.ERROR)
    yield ray
    ray.shutdown()


@pytest.fixture(params=["thread", pytest.param("ray", marks=pytest.mark.ray)])
def backend(request):
    if request.param == "ray":
        request.getfixturevalue("ray_session")
    return request.param


@pytest.fixture
def new_pool(backend):
    """Build pools for the backend under test and close them afterwards."""
    from rollout_fabric.pool import make_pool

    pools = []

    def make(cfg):
        pools.append(make_pool(backend, cfg))
        return pools[-1]

    yield make
    for pool in pools:
        pool.close()
