# -*- coding: utf-8 -*-
"""Shared fixtures for unit tests involving dolfin."""

# Copyright (C) 2014-2014 Martin Sandve Alnæs and Aslak Wigdahl Bergersen
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

from __future__ import print_function
import pytest
import os
import shutil
import tempfile
import gc
import platform
from instant import get_status_output
from dolfin import  *

from .fixtures import filedir, gc_barrier
from .skips import skip_in_parallel

@skip_in_parallel
@pytest.mark.cpp
def cpp_tester(request):
    gc_barrier()
    filedir = os.path.dirname(os.path.abspath(request.module.__file__))

    prefixes = ["", "mpirun -np 3 "]
    curdir = os.getcwd()
    os.chdir(filedir)
    try:
        # Set non-interactive, configure and build
        os.putenv('DOLFIN_NOPLOT', '1')

        #status, out_cmake = get_status_output('cmake .')
        #assert status == 0

        #status, out_make = get_status_output('make')
        #assert status == 0

        # Find all built test binaries
        list_of_tests = []
        for file in sorted(os.listdir('.')):
            if file.startswith("test_") and not '.' in file:
                list_of_tests.append(file)
        assert list_of_tests, "C++ tests have not been built!"

        # Run test binaries
        for prefix in prefixes: # With or without mpirun
            for test in list_of_tests:
                cmd = prefix + os.path.join('.', test)
                print("Running cmd: ", cmd)
                status, output = get_status_output(cmd)
                if not (status == 0 and 'OK' in output):
                    print(output)
                assert status == 0 and 'OK' in output

    finally:
        os.chdir(curdir)

    gc_barrier()
