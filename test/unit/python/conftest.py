import pytest
import gc
from dolfin import MPI, mpi_comm_world

def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""
    MPI.barrier(mpi_comm_world())
    gc.collect()
    MPI.barrier(mpi_comm_world())
