"Unit tests for XML input/output of MeshFunction (class XMLMeshFunction)"

# Copyright (C) 2011-2014 Garth N. Wells
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
#
# Modified by Anders Logg 2011

import pytest
import os
from dolfin import *
from dolfin_utils.test import skip_in_parallel, fixture, cd_tempdir


@skip_in_parallel
def test_io_size_t(cd_tempdir):
    "Test input/output for size_t"

    # Write some data
    mesh = UnitSquareMesh(5, 5)
    f = MeshFunction("size_t", mesh, 1)
    f.set_all(0)
    f[2] = 3
    f[5] = 7

    filename = "XMLMeshFunction_test_io_size_t.xml"

    # Write
    output_file = File(filename)
    output_file << f

    # Read from file
    g = MeshFunction("size_t", mesh, 1)
    input_file = File(filename)
    input_file >> g

    # Check values
    for i in range(f.size()):
        assert f[i] == g[i]


@skip_in_parallel
def test_io_int(cd_tempdir):
    "Test input/output for int"

    # Write some data
    mesh = UnitSquareMesh(5, 5)
    f = MeshFunction("int", mesh, 1)
    f.set_all(0)
    f[2] = -3
    f[5] = 7

    filename = "XMLMeshFunction_test_io_int.xml"

    # Write
    output_file = File(filename)
    output_file << f

    # Read from file
    g = MeshFunction("int", mesh, 1)
    input_file = File(filename)
    input_file >> g

    # Check values
    for i in range(f.size()):
        assert f[i] == g[i]


@skip_in_parallel
def test_io_double(cd_tempdir):
    "Test input/output for double"

    # Write some data
    mesh = UnitSquareMesh(5, 5)
    f = MeshFunction("double", mesh, 1)
    f.set_all(0.0)
    f[2] = 3.14
    f[5] = 10000000.0

    filename = "XMLMeshFunction_test_io_double.xml"

    # Write
    output_file = File(filename)
    output_file << f

    # Read from file
    g = MeshFunction("double", mesh, 1)
    input_file = File(filename)
    input_file >> g

    # Check values
    for i in range(f.size()):
        assert f[i] == g[i]


@skip_in_parallel
def test_io_bool(cd_tempdir):
    "Test input/output for bool"

    # Write some data
    mesh = UnitSquareMesh(5, 5)
    f = MeshFunction("bool", mesh, 1)
    f.set_all(False)
    f[2] = True
    f[5] = False

    filename = "XMLMeshFunction_test_io_bool.xml"

    # Write
    output_file = File(filename)
    output_file << f

    # Read from file
    g = MeshFunction("bool", mesh, 1)
    input_file = File(filename)
    input_file >> g

    # Check values
    for i in range(f.size()):
        assert f[i] == g[i]
