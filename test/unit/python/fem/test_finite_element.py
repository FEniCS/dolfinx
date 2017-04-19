#!/usr/bin/env py.test

"""Unit tests for the fem interface"""

# Copyright (C) 2009 Garth N. Wells
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
# First added:  2009-07-28
# Last changed: 2009-07-28

import pytest
import numpy
from dolfin import *
from six.moves import xrange as range
from dolfin_utils.test import fixture


@fixture
def mesh():
    return UnitSquareMesh(4, 4)


@fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@fixture
def Q(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)


@fixture
def W(mesh):
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    return FunctionSpace(mesh, V*Q)


def test_evaluate_dofs(W, mesh, V):

    e = Expression("x[0] + x[1]", degree=1)
    e2 = Expression(("x[0] + x[1]", "x[0] + x[1]"), degree=1)

    coords = numpy.zeros((3, 2), dtype="d")
    coord = numpy.zeros(2, dtype="d")
    values0 = numpy.zeros(3, dtype="d")
    values1 = numpy.zeros(3, dtype="d")
    values2 = numpy.zeros(3, dtype="d")
    values3 = numpy.zeros(3, dtype="d")
    values4 = numpy.zeros(6, dtype="d")

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    for cell in cells(mesh):
        vx = cell.get_vertex_coordinates()
        orientation = cell.orientation()
        V.element().tabulate_dof_coordinates(cell, coords)
        for i in range(coords.shape[0]):
            coord[:] = coords[i, :]
            values0[i] = e(*coord)
        L0.element().evaluate_dofs(values1, e, vx, orientation, cell)
        L01.element().evaluate_dofs(values2, e, vx, orientation, cell)
        L11.element().evaluate_dofs(values3, e, vx, orientation, cell)
        L1.element().evaluate_dofs(values4, e2, vx, orientation, cell)

        for i in range(3):
            assert round(values0[i] - values1[i], 7) == 0
            assert round(values0[i] - values2[i], 7) == 0
            assert round(values0[i] - values3[i], 7) == 0
            assert round(values4[:3][i] - values0[i], 7) == 0
            assert round(values4[3:][i] - values0[i], 7) == 0


def test_evaluate_dofs_manifolds_affine():
    "Testing evaluate_dofs vs tabulated coordinates."

    n = 4
    mesh = BoundaryMesh(UnitSquareMesh(n, n), "exterior")
    mesh2 = BoundaryMesh(UnitCubeMesh(n, n, n), "exterior")
    DG0 = FunctionSpace(mesh, "DG", 0)
    DG1 = FunctionSpace(mesh, "DG", 1)
    CG1 = FunctionSpace(mesh, "CG", 1)
    CG2 = FunctionSpace(mesh, "CG", 2)
    DG20 = FunctionSpace(mesh2, "DG", 0)
    DG21 = FunctionSpace(mesh2, "DG", 1)
    CG21 = FunctionSpace(mesh2, "CG", 1)
    CG22 = FunctionSpace(mesh2, "CG", 2)
    elements = [DG0, DG1, CG1, CG2, DG20, DG21, CG21, CG22]

    f = Expression("x[0] + x[1]", degree=1)
    for V in elements:
        sdim = V.element().space_dimension()
        gdim = V.mesh().geometry().dim()
        coords = numpy.zeros((sdim, gdim), dtype="d")
        coord = numpy.zeros(gdim, dtype="d")
        values0 = numpy.zeros(sdim, dtype="d")
        values1 = numpy.zeros(sdim, dtype="d")
        for cell in cells(V.mesh()):
            vx = cell.get_vertex_coordinates()
            orientation = cell.orientation()
            V.element().tabulate_dof_coordinates(cell, coords)
            for i in range(coords.shape[0]):
                coord[:] = coords[i, :]
                values0[i] = f(*coord)
            V.element().evaluate_dofs(values1, f, vx, orientation, cell)
            for i in range(sdim):
                assert round(values0[i] - values1[i], 7) == 0


def test_tabulate_coord(V, W, mesh):

    coord0 = numpy.zeros((3, 2), dtype="d")
    coord1 = numpy.zeros((3, 2), dtype="d")
    coord2 = numpy.zeros((3, 2), dtype="d")
    coord3 = numpy.zeros((3, 2), dtype="d")
    coord4 = numpy.zeros((6, 2), dtype="d")

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    for cell in cells(mesh):
        V.element().tabulate_dof_coordinates(cell, coord0)
        L0.element().tabulate_dof_coordinates(cell, coord1)
        L01.element().tabulate_dof_coordinates(cell, coord2)
        L11.element().tabulate_dof_coordinates(cell, coord3)
        L1.element().tabulate_dof_coordinates(cell, coord4)

        assert (coord0 == coord1).all()
        assert (coord0 == coord2).all()
        assert (coord0 == coord3).all()
        assert (coord4[:3] == coord0).all()
        assert (coord4[3:] == coord0).all()
