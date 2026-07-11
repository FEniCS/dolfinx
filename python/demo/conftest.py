"""Configuration for demo tests."""

import os
import tempfile

import pytest


def pytest_configure(config):
    """Set a per-worker matplotlib config/cache directory and rc file.

    The rc overrides disable Unicode minus-sign rendering (falls back
    to plain ASCII hyphen-minus) and glyph hinting, which sidestep a
    FreeType rasterizer bug ("raster overflow") triggered by certain
    glyph outlines at certain sizes during savefig() in some demo
    tests.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    cache_dir = os.path.join(tempfile.gettempdir(), f"mplconfig-{worker_id}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir

    rc_path = os.path.join(cache_dir, "matplotlibrc")
    with open(rc_path, "w") as f:
        f.write("axes.unicode_minus : False\n")
        f.write("text.hinting : none\n")
    os.environ["MATPLOTLIBRC"] = rc_path


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
