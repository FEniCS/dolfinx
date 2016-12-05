#!/usr/bin/env py.test

"""Unit tests for PointSources"""

# Copyright (C) 2011-2012 Ettie Unwin
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
import numpy as np
from dolfin import *
parameters["ghost_mode"] = "shared_facet"

def test_pointsource_vector_node():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector """
    data = [[UnitIntervalMesh(10), Point(0.5), 5],
            [UnitSquareMesh(2,2), Point(0.5, 0.5), 4],
            [UnitCubeMesh(2,2,2), Point(0.5, 0.5, 0.5), 6]]

    for dim in range(1):
        mesh = data[dim][0]
        point = data[dim][1]
        g_indices = data[dim][2]
        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)
        print b.array()
        b_sum = MPI.sum(mesh.mpi_comm(), np.sum(b.array()))
        assert b_sum == pytest.approx(10.0)
        assert b.array()[data[dim][2]] == pytest.approx(10.0)

def test_pointsource_vector():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector that isn't placed at a node for 1D, 2D and 3D. """
    data = [[UnitIntervalMesh(10), Point(0.05), [9,10]],
            [UnitSquareMesh(1,1), Point(2.0/3.0, 1.0/3.0), [1,2,3]],
            [UnitCubeMesh(1,1,1), Point(2.0/3.0, 1.0/3.0, 1.0/3.0), [1,2,5]]]

    for dim in range(1):
        mesh = data[dim][0]
        point = data[dim][1]
        g_indices = data[dim][2]

        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)
        print b.str(True)
        b_sum = MPI.sum(mesh.mpi_comm(), np.sum(b.array()))
        assert b_sum == pytest.approx(10.0)


        n = len(g_indices)
        for i in range(n):
            assert b.array()[g_indices[i]] == pytest.approx(10.0/n)

test_pointsource_vector_node()
