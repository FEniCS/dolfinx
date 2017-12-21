"""Unit tests for the XML io library for vectors"""

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

import pytest
from dolfin import *
import os
from dolfin_utils.test import skip_if_not_PETSc, fixture, cd_tempdir


@skip_if_not_PETSc
def test_save_vector_petsc(cd_tempdir):
    # Create vector and write file
    x = PETScVector(MPI.comm_world, 197)
    x[:] = 1.0
    f = File("x.xml")
    f << x

@skip_if_not_PETSc
def test_save_gzipped_vector(cd_tempdir):
    # Create vector and write file
    x = PETScVector(MPI.comm_world, 197)
    x[:] = 1.0
    f = File("x.xml.gz")
    f << x

@skip_if_not_PETSc
def test_read_vector_petcs(cd_tempdir):
    # Create vector and write file
    x = PETScVector(MPI.comm_world, 197)
    x[:] = 1.0
    f = File("x.xml")
    f << x

    # Read vector from previous write
    y = PETScVector()
    f >> y
    assert x.size() == y.size()
    assert round(x.norm("l2") - y.norm("l2"), 7) == 0


@skip_if_not_PETSc
def test_read_gzipped_vector(cd_tempdir):
    # Create vector and write file
    x = PETScVector(MPI.comm_world, 197)
    x[:] = 1.0
    f = File("x.xml")
    f << x

    # Read vector from previous write
    y = PETScVector()
    f >> y
    assert x.size() == y.size()
    assert round(x.norm("l2") - y.norm("l2"), 7) == 0

def test_save_read_vector(cd_tempdir):
    size = 512
    x = Vector(MPI.comm_world, size)
    x[:] = 1.0

    out_file = File("test_vector_xml.xml")
    out_file << x

    y = Vector()
    out_file >> y
    assert x.size() == y.size()
    assert round((x - y).norm("l2") - 0.0, 7) == 0
