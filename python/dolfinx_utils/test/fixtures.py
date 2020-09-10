# Copyright (C) 2014-2014 Martin Sandve Alnæs and Aslak Wigdahl Bergersen
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Shared fixtures for unit tests."""

import os
import time
import shutil
from collections import defaultdict
from mpi4py import MPI

import pytest

# --- Test fixtures (use as is or as examples): ---


def worker_id(request):
    """Returns thread id when running with pytest-xdist in parallel."""
    try:
        return request.config.slaveinput['slaveid']
    except AttributeError:
        return 'master'


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

        # Wait until the above created the directory
        waited = 0
        while not os.path.exists(path):
            time.sleep(0.1)
            waited += 0.1

            if waited > 1:
                raise RuntimeError(f"Unable to create test directory {path}")

    comm.Barrier()

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

    MPI safe (assuming MPI.COMM_WORLD context).

    """
    return _create_tempdir(request)


# @pytest.yield_fixture(scope="function")
# def cd_tempdir(request):
#     """Return a unique directory name for this test function instance.

#     Deletes and re-creates directory from previous test runs but lets
#     the directory stay after the test run for eventual inspection.

#     Returns the directory name, derived from the test file and
#     function plus a sequence number to work with parameterized tests.

#     Changes the current directory to the tempdir and resets cwd
#     afterwards.

#     MPI safe (assuming MPI.COMM_WORLD context).

#     """
#     cwd = os.getcwd()
#     path = _create_tempdir(request)
#     os.chdir(path)
#     yield path
#     os.chdir(cwd)
