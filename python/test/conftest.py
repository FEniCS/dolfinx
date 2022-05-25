import gc
import os
import shutil
import time
from collections import defaultdict

import pytest

from mpi4py import MPI


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
    comm = MPI.COMM_WORLD
    comm.Barrier()


# Add 'skip_in_parallel' skip
def pytest_runtest_setup(item):
    marker = item.get_closest_marker("skip_in_parallel")
    if marker and MPI.COMM_WORLD.size > 1:
        pytest.skip("This test should only be run in serial")


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


def _worker_id(request):
    """Returns thread id when running with pytest-xdist in parallel."""
    try:
        return request.config.workerinput["workerid"]
    except AttributeError:
        return "master"


def _create_tempdir(request):
    # Get directory name of test_foo.py file
    testfile = request.module.__file__
    testfiledir = os.path.dirname(os.path.abspath(testfile))

    # Construct name test_foo_tempdir from name test_foo.py
    testfilename = os.path.basename(testfile)
    outputname = testfilename.replace(".py", "_tempdir_{}".format(
        _worker_id(request)))

    # Get function name test_something from test_foo.py
    function = request.function.__name__

    # Join all of these to make a unique path for this test function
    basepath = os.path.join(testfiledir, outputname)
    path = os.path.join(basepath, function)

    # Add a sequence number to avoid collisions when tests are otherwise
    # parameterized
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        _create_tempdir._sequencenumber[path] += 1
        sequencenumber = _create_tempdir._sequencenumber[path]
    else:
        sequencenumber = None

    sequencenumber = comm.bcast(sequencenumber)
    path += "__" + str(sequencenumber)

    # Delete and re-create directory on root node
    if comm.rank == 0:
        # First time visiting this basepath, delete the old and create a
        # new
        if basepath not in _create_tempdir._basepaths:
            _create_tempdir._basepaths.add(basepath)
            if os.path.exists(basepath):
                shutil.rmtree(basepath)
            # Make sure we have the base path test_foo_tempdir for this
            # test_foo.py file
            if not os.path.exists(basepath):
                os.mkdir(basepath)

        # Delete path from old test run
        if os.path.exists(path):
            shutil.rmtree(path)
        # Make sure we have the path for this test execution: e.g.
        # test_foo_tempdir/test_something__3
        if not os.path.exists(path):
            os.mkdir(path)

        # Wait until the above created the directory
        waited = 0
        while not os.path.exists(path):
            time.sleep(0.1)
            waited += 0.1

            if waited > 1:
                raise RuntimeError(f"Unable to create test directory {path}")

    comm.Barrier()

    return path


# Assigning a function member variables is a bit of a nasty hack
_create_tempdir._sequencenumber = defaultdict(int)  # type: ignore
_create_tempdir._basepaths = set()  # type: ignore


@pytest.fixture(scope="function")
def tempdir(request):
    """Return a unique directory name for this test function instance.

    Deletes and re-creates directory from previous test runs but lets
    the directory stay after the test run for eventual inspection.

    Returns the directory name, derived from the test file and
    function plus a sequence number to work with parameterized tests.

    Does NOT change the current directory.

    MPI safe (assuming MPI.COMM_WORLD context).

    """
    return _create_tempdir(request)
