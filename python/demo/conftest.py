import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mpiexec",
        action="store",
        default="mpiexec",
        help="Name of program to run MPI, e.g. mpiexec",
    )
    parser.addoption("--num-proc", action="store", default=1, help="Number of MPI processes to use")


@pytest.fixture
def mpiexec(request):
    return request.config.getoption("--mpiexec")


@pytest.fixture
def num_proc(request):
    return request.config.getoption("--num-proc")
