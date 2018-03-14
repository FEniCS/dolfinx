"""Shared fixtures for unit tests involving dolfin."""

# Copyright (C) 2014-2014 Martin Sandve Alnæs and Aslak Wigdahl Bergersen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import pytest
import os
import shutil
import tempfile
import gc
import platform
import decorator

from dolfin.parameter import  parameters
from dolfin import *

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
        # and return collectivelly
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


@pytest.fixture
def worker_id(request):
    """Returns thread id when running with pytest-xdist in parallel."""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
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


@pytest.fixture(params=[False, True])
def true_false_fixture(request):
    "A fixture setting the values true and false."
    gc_barrier()
    return request.param


@pytest.fixture(scope="module")
def filedir(request):
    "Return the directory of the test module."
    gc_barrier()
    d = os.path.dirname(os.path.abspath(request.module.__file__))
    return d


@pytest.fixture(scope="module")
def rootdir(request):
    """Return the root directory of the repository. Assumes run from
    within repository filetree.

    """
    gc_barrier()
    d = os.path.dirname(os.path.abspath(request.module.__file__))
    t = ''
    while t != "test":
        d, t = os.path.split(d)
    return d


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


def _create_tempdir(request):
    # Get directory name of test_foo.py file
    testfile = request.module.__file__
    testfiledir = os.path.dirname(os.path.abspath(testfile))

    # Construct name test_foo_tempdir from name test_foo.py
    testfilename = os.path.basename(testfile)
    outputname = testfilename.replace(".py",
                                      "_tempdir_{}".format(worker_id(request)))

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
from collections import defaultdict
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


@pytest.yield_fixture
def pushpop_parameters():
    global parameters
    gc_barrier()
    prev = parameters.copy()
    yield parameters.copy()
    parameters.assign(prev)


# TODO: Rename set_parameters_fixture to e.g. use_parameter_values
def set_parameters_fixture(paramname, values, key=lambda x: x):
    """Return a fixture that sets and resets a global parameter to each of
    a list of values before and after each test run.  Allows
    paramname="foo.bar.var" meaning parameters["foo"]["bar"]["var"].

    Usage:
        repr = set_parameters_fixture("form_compiler.representation", ["quadrature", "uflacs"])
        my_fixture1 = set_parameters_fixture("linear_algebra_backend", ["PETSc"])
        my_fixture2 = set_parameters_fixture("linear_algebra_backend", [("Eigen", "")], key=lambda x: x[0])

        def test_something0(repr):
            assert repr in ("quadrature", "uflacs")
            assert parameters["form_compiler"]["representation"] == repr

        def test_something1(my_fixture1):
            assert my_fixture1 in ("PETSc")
            assert parameters["linear_algebra_backend"] == my_fixture1

        def test_something2(my_fixture2):
            assert my_fixture2[0] in ("Eigen")
            assert parameters["linear_algebra_backend"] == my_fixture2[0]

    Try it and see.

    """
    global parameters
    def _pushpop(request):
        gc_barrier()
        if '.' in paramname:
            names = paramname.split('.')
            if len(names) == 2:
                prev = parameters[names[0]][names[1]]                # Remember original value
                parameters[names[0]][names[1]] = key(request.param)  # Set value
                yield request.param                                  # Let test run
                parameters[names[0]][names[1]] = prev                # Reset value
            elif len(names) == 3:
                prev = parameters[names[0]][names[1]][names[2]]                # Remember original value
                parameters[names[0]][names[1]][names[2]] = key(request.param)  # Set value
                yield request.param                                            # Let test run
                parameters[names[0]][names[1]][names[2]] = prev                # Reset value
        else:
            prev = parameters[paramname]                # Remember original value
            parameters[paramname] = key(request.param)  # Set value
            yield request.param                         # Let test run
            parameters[paramname] = prev                # Reset value

    return pytest.yield_fixture(scope="function", params=values)(_pushpop)
