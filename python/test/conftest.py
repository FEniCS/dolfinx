import gc
import os
import shutil
import time
from collections import defaultdict
from scipy.sparse.linalg import spsolve
from dolfinx.la import vector as dolfinx_vector
import numpy as np

from mpi4py import MPI

import pytest


def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""

    # Do the normal teardown
    item.teardown()

    # Collect the garbage (call destructors collectively)
    del item
    # NOTE: How are we sure that 'item' does not hold references to
    # temporaries and someone else does not hold a reference to 'item'?!
    # Well, it seems that it works...
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
    within repository filetree."""
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
    outputname = testfilename.replace(".py", f"_tempdir_{_worker_id(request)}")

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


def _global_dot(comm, v0, v1, n):
    local_dot = np.dot(v0[:n], v1[:n])
    return comm.allreduce(local_dot, MPI.SUM)


def _cg(comm, A0, b, x):
    kmax = 500
    rtol = 1e-8

    A = A0.to_scipy()
    nr = A.shape[0]
    assert nr == A0.index_map(0).size_local

    x.scatter_forward()
    y = A @ x.array
    r = b[:nr] - y
    p = dolfinx_vector(A0.index_map(1), dtype=x.array.dtype)
    p.array[:nr] = r

    # Iterations of CG
    rnorm0 = _global_dot(comm, r, r, nr)
    rtol2 = rtol * rtol
    rnorm = rnorm0
    k = 0
    while (k < kmax):
        k += 1
        p.scatter_forward()
        y = A @ p.array
        alpha = rnorm / _global_dot(comm, p.array, y, nr)
        x.array[:] += alpha * p.array
        r -= alpha * y
        rnorm_new = _global_dot(comm, r, r, nr)
        beta = rnorm_new / rnorm
        rnorm = rnorm_new
        if (rnorm / rnorm0 < rtol2):
            break
        p.array[:nr] = beta * p.array[:nr] + r

    return k, x


@pytest.fixture(scope="function")
def solver():

    def _solve(comm, A, b):
        if comm.size == 1:
            AA = A.to_scipy()
            bb = b.array
            return spsolve(AA, bb)
        else:
            x = dolfinx_vector(A.index_map(1), dtype=b.array.dtype)
            x.array[:] = 0.0
            _cg(comm, A, b.array, x)
            return x.array[:len(b.array)]

    return _solve
