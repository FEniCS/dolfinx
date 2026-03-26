import pytest


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
