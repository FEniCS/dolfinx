#!/usr/bin/env py.test

# Copyright (C) Aslak Wigdahl Bergersen
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import os
import pytest
import platform
from instant import get_status_output

from dolfin_utils.test import skip_in_parallel

def pytest_generate_tests(metafunc):
    # Skip in parallel (the C++ test should be run in parallel but not the pytest python process)
    from dolfin import MPI, mpi_comm_world
    if MPI.size(mpi_comm_world()) > 1:
        return

    # Set non-interactive
    os.putenv('DOLFIN_NOPLOT', '1')

    curdir = os.getcwd()
    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    out_cmake = os.system('cmake .')
    out_make = os.system('make')

    #Find all tests
    list_of_tests = []
    for file in os.listdir('.'):
        if file.startswith("test_") and not '.' in file:
            if platform.system() == 'Windows':
                file += '.exe'
            list_of_tests.append(file)

    # Create tests
    os.chdir(curdir)
    funcargs = {}
    funcargs['curdir'] = curdir
    funcargs['filepath'] = filepath
    for test in list_of_tests:
        funcargs['test'] = test
        metafunc.addcall(funcargs=funcargs)

@skip_in_parallel
@pytest.mark.cpp
def test_cpp_mesh(test, curdir, filepath):
    os.chdir(filepath)
    status, output = get_status_output(os.path.join('.', test))
    assert 'OK' in output
    assert status == 0
    os.chdir(curdir)
