#!/usr/bin/env py.test

"""Unit tests for PointSources"""

# Copyright (C) 2016 Ettie Unwin
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

def test_pointsource_vector_node():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector and when placed at a node for 1D, 2D and 3D. """
    data = [[UnitIntervalMesh(10), Point(0.5)],
            [UnitSquareMesh(10,10), Point(0.5, 0.5)],
            [UnitCubeMesh(3,3,3), Point(0.5, 0.5, 0.5)]]

    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)

        # Checks array sums to correct value
        b_sum = b.sum()
        assert round(b_sum - 10.0) == 0

        # Checks point source is added to correct part of the array
        v2d = vertex_to_dof_map(V)
        for v in vertices(mesh):
            if near(v.midpoint().distance(point), 0.0):
                ind = v2d[v.index()]
                if ind<len(b.array()):
                    assert round(b.array()[ind] - 10.0) == 0

def test_pointsource_vector():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector that isn't placed at a node for 1D, 2D and 3D. """
    data = [UnitIntervalMesh(10), UnitSquareMesh(10,10), UnitCubeMesh(3,3,3)]

    for dim in range(3):
        mesh = data[dim]

        cell = Cell(mesh, 0)
        point = cell.midpoint()

        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(Constant(0.0)*v*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)

        # Checks array sums to correct value
        b_sum = b.sum()
        assert round(b_sum - 10.0) == 0


def test_pointsource_vector_fs():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector for a vector function space that isn't placed at a node for
    1D, 2D and 3D. """
    data = [[UnitIntervalMesh(10), Point(0.5)],
            [UnitSquareMesh(10,10), Point(0.5, 0.5)],
            [UnitCubeMesh(10,10,10), Point(0.5, 0.5, 0.5)]]

    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        V = VectorFunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        b = assemble(dot(Constant([0.0]*mesh.geometry().dim()),v)*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)

        # Checks array sums to correct value
        b_sum = b.sum()
        assert round(b_sum - 10.0*V.num_sub_spaces()) == 0

        # Checks point source is added to correct part of the array
        v2d = vertex_to_dof_map(V)
        for v in vertices(mesh):
            if near(v.midpoint().distance(point), 0.0):
                for spc_idx in range(V.num_sub_spaces()):
                    ind = v2d[v.index()*V.num_sub_spaces() + spc_idx]
                    if ind<len(b.array()):
                        assert round(b.array()[ind] - 10.0) == 0


def test_pointsource_mixed_space():
    """Tests point source when given constructor PointSource(V, point, mag)
    with a vector for a mixed function space that isn't placed at a node for
    1D, 2D and 3D. """
    data = [[UnitIntervalMesh(10), Point(0.5)],
            [UnitSquareMesh(10,10), Point(0.5, 0.5)],
            [UnitCubeMesh(3,3,3), Point(0.5, 0.5, 0.5)]]

    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        ele1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        ele2 = FiniteElement("DG", mesh.ufl_cell(), 2)
        ele3 = VectorElement("CG", mesh.ufl_cell(), 2)
        V = FunctionSpace(mesh, MixedElement([ele1, ele2, ele3]))
        value_dimension = V.element().value_dimension(0)
        v = TestFunction(V)
        b = assemble(dot(Constant([0.0]*value_dimension),v)*dx)
        ps = PointSource(V, point, 10.0)
        ps.apply(b)

        # Checks array sums to correct value
        b_sum = b.sum()
        assert round(b_sum - 10.0*value_dimension) == 0

def test_point_outside():
    mesh = UnitIntervalMesh(10)
    point = Point(1.2)
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)
    ps = PointSource(V, point, 10.0)

    print 'here'
    #with pytest.raises(RuntimeError):
    #ps.apply(b)

def test_pointsource_matrix():
    data = [[UnitIntervalMesh(10), Point(0.5)],
            [UnitSquareMesh(2,2), Point(0.5, 0.5)],
            [UnitCubeMesh(2,2,2), Point(0.5, 0.5, 0.5)]]

    for dim in range(3):
        mesh = data[dim][0]
        point = data[dim][1]
        V = FunctionSpace(mesh, "CG", 1)

        u, v = TrialFunction(V), TestFunction(V)
        w = Function(V)
        A = assemble(Constant(0.0)*u*v*dx)
        ps = PointSource(V, V, point, 10.0)
        ps.apply(A)

        # Checks array sums to correct value
        a_sum =  MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
        assert round(a_sum - 10.0) == 0

        # Checks point source is added to correct part of the array
        A.get_diagonal(w.vector())
        v2d = vertex_to_dof_map(V)
        for v in vertices(mesh):
            if near(v.midpoint().distance(point), 0.0):
                ind = v2d[v.index()]
                if ind<len(A.array()):
                    assert round(w.vector()[ind] - 10.0) == 0
                    info("Asserted")

test_point_outside()
