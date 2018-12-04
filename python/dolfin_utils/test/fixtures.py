# Copyright (C) 2014-2014 Martin Sandve Alnæs and Aslak Wigdahl Bergersen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Shared fixtures for unit tests."""

import gc
import os
import shutil
from collections import defaultdict

import decorator
import pytest

from dolfin import MPI

# --- Test fixtures (use as is or as examples): ---


def fixture(func):
    """Decorator for creating module scope fixture.
    Also take care of garbage collection of temporaries.

    NOTE: Probably does not work with yield fixtures or
    maybe just ignores post-yield code

    This is the preferred decorator for writing fixtures involving
    objects which might have collective destructors.  If in a need for
    using ``pytest.fixture`` directly, do::

        yield result_of_fixture

        gc.collect()
        MPI.barrier(MPI.comm_world)

    """

    def wrapper(func, *args, **kwargs):

        # Run function
        rv = func(*args, **kwargs)

        # Collect garbage of temporaries of the function
        # and return collectively
        gc.collect()
        MPI.barrier(MPI.comm_world)

        return rv

        # FIXME: Need also to find a way how to clean-up fixture
        #        return value; yield does not work here

    # Decorate function with the wrapper
    wrapped = decorator.decorator(wrapper, func)

    # Mark as fixture and return
    return pytest.fixture(scope='module')(wrapped)


def gc_barrier():
    """Internal utility to easily switch on and off calls to gc.collect()
    and MPI.barrier(world) in all fixtures here.  Helps make the tests
    deterministic when debugging.

    """
    gc.collect()
    if MPI.size(MPI.comm_world) > 1:
        MPI.barrier(MPI.comm_world)


def worker_id(request):
    """Returns thread id when running with pytest-xdist in parallel."""
    try:
        return request.config.slaveinput['slaveid']
    except AttributeError:
        return 'master'


@pytest.yield_fixture(scope="function")
def gc_barrier_fixture():
    """Function decorator to call gc.collect() and
    MPI.barrier(world) before and after a test.  Helps
    make the tests deterministic when debugging.

    NOTE: This decorator is not needed now for writing tests, as there
    is ``gc.collect()`` call in ``conftest.py`` on teardown of every
    test.

    """
    gc_barrier()
    yield
    gc_barrier()


use_gc_barrier = pytest.mark.usefixtures("gc_barrier_fixture")


def _create_tempdir(request):
    # Get directory name of test_foo.py file
    testfile = request.module.__file__
    testfiledir = os.path.dirname(os.path.abspath(testfile))

    # Construct name test_foo_tempdir from name test_foo.py
    testfilename = os.path.basename(testfile)
    outputname = testfilename.replace(".py", "_tempdir_{}".format(
        worker_id(request)))

    # Get function name test_something from test_foo.py
    function = request.function.__name__

    # Join all of these to make a unique path for this test function
    basepath = os.path.join(testfiledir, outputname)
    path = os.path.join(basepath, function)

    # Add a sequence number to avoid collisions when tests are
    # otherwise parameterized
    if MPI.rank(MPI.comm_world) == 0:
        _create_tempdir._sequencenumber[path] += 1
        sequencenumber = _create_tempdir._sequencenumber[path]
        sequencenumber = MPI.sum(MPI.comm_world, sequencenumber)
    else:
        sequencenumber = MPI.sum(MPI.comm_world, 0)
    path += "__" + str(sequencenumber)

    # Delete and re-create directory on root node
    if MPI.rank(MPI.comm_world) == 0:
        # First time visiting this basepath, delete the old and create
        # a new
        if basepath not in _create_tempdir._basepaths:
            _create_tempdir._basepaths.add(basepath)
            if os.path.exists(basepath):
                shutil.rmtree(basepath)
            # Make sure we have the base path test_foo_tempdir for
            # this test_foo.py file
            if not os.path.exists(basepath):
                os.mkdir(basepath)

        # Delete path from old test run
        if os.path.exists(path):
            shutil.rmtree(path)
        # Make sure we have the path for this test execution:
        # e.g. test_foo_tempdir/test_something__3
        if not os.path.exists(path):
            os.mkdir(path)
    MPI.barrier(MPI.comm_world)

    return path


_create_tempdir._sequencenumber = defaultdict(int)
_create_tempdir._basepaths = set()


@pytest.fixture(scope="function")
def tempdir(request):
    """Return a unique directory name for this test function instance.

    Deletes and re-creates directory from previous test runs but lets
    the directory stay after the test run for eventual inspection.

    Returns the directory name, derived from the test file and
    function plus a sequence number to work with parameterized tests.

    Does NOT change the current directory.

    MPI safe (assuming MPI.comm_world context).

    """
    gc_barrier()
    return _create_tempdir(request)


@pytest.yield_fixture(scope="function")
def cd_tempdir(request):
    """Return a unique directory name for this test function instance.

    Deletes and re-creates directory from previous test runs but lets
    the directory stay after the test run for eventual inspection.

    Returns the directory name, derived from the test file and
    function plus a sequence number to work with parameterized tests.

    Changes the current directory to the tempdir and resets cwd
    afterwards.

    MPI safe (assuming MPI.comm_world context).

    """
    gc_barrier()
    cwd = os.getcwd()
    path = _create_tempdir(request)
    os.chdir(path)
    yield path
    os.chdir(cwd)
