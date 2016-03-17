#!/usr/bin/env py.test

"""Unit tests for TimeSeries"""

# Copyright (C) 2011-2014 Marie E. Rognes
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
from dolfin import *
import os
from dolfin_utils.test import skip_in_parallel, skip_if_not_HDF5, fixture, tempdir

@skip_in_parallel
@skip_if_not_HDF5
def test_retrieve_all_connectivities(tempdir):
    _test_retrieve(tempdir, "test_retrieve_all_connectivities")

def _test_retrieve(tempdir, basename):
    filename = os.path.join(tempdir, basename)

    times = [t/10.0 for t in range(1, 11)]

    mesh_size = (2, 2, 2)
    mesh = UnitCubeMesh(*mesh_size)
    mesh.init()
    V = FunctionSpace(mesh, "CG", 2)

    u = Function(V)
    series = TimeSeries(mpi_comm_world(), filename)
    for t in times:
        u.vector()[:] = t
        series.store(u.vector(), t)
        series.store(mesh, t)

    series = TimeSeries(mpi_comm_world(), filename)
    t0 = series.vector_times()[0]
    T = series.mesh_times()[-1]

    assert round(t0 - times[0], 7) == 0
    assert round(T - times[-1], 7) == 0

    # Test retreiving of mesh
    mesh_retreived = Mesh()
    series.retrieve(mesh_retreived, 0.1)

    mesh_test = mesh

    for entity in range(4):
        mesh_retreived.init(entity)
        assert mesh_retreived.topology().size(entity) == \
            mesh_test.topology().size(entity)

    for i in range(4):
        for j in range(4):
            mesh_retreived.init(i, j)
            assert mesh_retreived.topology()(i, j).size() == \
                              mesh_test.topology()(i, j).size()


@skip_in_parallel
@skip_if_not_HDF5
def test_subdirectory(tempdir):
    "Test that retrieve/store works with nonexisting subdirectory"
    filename = os.path.join(tempdir, "test_subdirectory")

    m0 = UnitSquareMesh(3, 3)

    series0 = TimeSeries(mpi_comm_world(), filename)
    x0 = Vector(mpi_comm_world(), 10)

    # Test storage of only one time point for the mesh
    series0.store(m0, 0.1)
    series0.store(x0, 0.1)
    series0.store(x0, 0.2)

    series1 = TimeSeries(mpi_comm_world(), filename)
    m1 = Mesh()
    x1 = Vector()

    series1.retrieve(m1, 0.1)
    series1.retrieve(x1, 0.15)
