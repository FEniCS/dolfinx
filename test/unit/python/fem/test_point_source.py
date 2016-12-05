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
#parameters["ghost_mode"] = "shared_facet"

def test_pointsource_vector_node():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector """
    data = [[UnitIntervalMesh(10), Point(0.5)],
            [UnitSquareMesh(10,10), Point(0.5, 0.5)],
            [UnitCubeMesh(10,10,10), Point(0.5, 0.5, 0.5)]]

    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)
        print b.array()

        b_sum = MPI.sum(mesh.mpi_comm(), np.sum(b.array()))
        assert b_sum == pytest.approx(10.0)

        v2d = vertex_to_dof_map(V)
        for v in vertices(mesh):
            if near(v.midpoint().distance(point), 0.0):
                ind = v2d[v.index()]
                if ind<len(b.array()):
                    assert b.array()[ind] == pytest.approx(10.0)

def test_pointsource_vector():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector that isn't placed at a node for 1D, 2D and 3D. """
    data = [[UnitIntervalMesh(10), Point(0.05), [0.05]],
            [UnitSquareMesh(10,10), Point(0.2/3.0, 0.1/3.0),
             [np.sqrt(2*(0.1/3.0)**2), np.sqrt((0.1/3.0)**2+(0.2/3.0)**2)]],
            [UnitCubeMesh(1,1,1), Point(3.0/4.0, 1.0/2.0, 1.0/4.0),
             [np.sqrt((3.0/4.0)**2 + (1.0/2.0)**2 + (1.0/4.0)**2),
              np.sqrt((1.0/2.0)**2 + 2*(1.0/4.0)**2)]]]


    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        length = data[dim][2]

        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)
        b_sum = MPI.sum(mesh.mpi_comm(), np.sum(b.array()))
        assert b_sum == pytest.approx(10.0)

        print b.array()
        v2d = vertex_to_dof_map(V)
        for i in range(len(length)):
            for v in vertices(mesh):
                if near(v.midpoint().distance(point), length[i]):
                    ind = v2d[v.index()]
                    print ind
                    print b.array()[ind]
                    #assert b.array()[ind] == pytest.approx(10.0/(mesh.geometry().dim()+1))

test_pointsource_vector()
