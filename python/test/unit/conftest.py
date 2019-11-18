import gc
from dolfin import MPI
import pytest
import os


def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""

    # Do the normal teardown
    item.teardown()

    # Collect the garbage (call destructors collectively)
    del item
    # NOTE: How are we sure that 'item' does not hold references
    #       to temporaries and someone else does not hold a reference
    #       to 'item'?! Well, it seems that it works...
    gc.collect()
    MPI.barrier(MPI.comm_world)


@pytest.fixture(scope="module")
def datadir(request):
    """Return the directory of the shared test data. Assumes run from
    within repository filetree.

    """
    d = os.path.dirname(os.path.abspath(request.module.__file__))
    t = os.path.join(d, "data")
    while not os.path.isdir(t):
        d, t = os.path.split(d)
        t = os.path.join(d, "data")
    return t
