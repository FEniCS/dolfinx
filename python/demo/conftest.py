"""Configuration for demo tests."""

import os
import tempfile

import pytest


def pytest_configure(config):
    """Set a per-worker matplotlib config/cache directory.

    This avoids concurrent pytest-xdist workers racing on the shared
    font cache, which otherwise produces intermittent FreeType errors
    like "raster overflow" during savefig() in parallel demo tests.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    cache_dir = os.path.join(tempfile.gettempdir(), f"mplconfig-{worker_id}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--mpiexec",
        action="store",
        default="mpiexec",
        help="Name of program to run MPI, e.g. mpiexec",
    )
    parser.addoption("--num-proc", action="store", default=1, help="Number of MPI processes to use")


@pytest.fixture
def mpiexec(request):
    """Name of program to run MPI, e.g. mpiexec."""
    return request.config.getoption("--mpiexec")


@pytest.fixture
def num_proc(request):
    """Number of MPI processes to use."""
    return request.config.getoption("--num-proc")
