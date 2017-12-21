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
from dolfin_utils.test import fixture


xfail = pytest.mark.xfail(strict=True)


@pytest.mark.parametrize('mesh_factory', [(UnitSquareMesh, (4, 4)),
                                          (UnitCubeMesh, (2, 2, 2)),
                                          (UnitSquareMesh.create, (4, 4, CellType.Type.quadrilateral)),
                                          # cell_normal has not been implemented for hex cell
                                          # cell.orientation() does not work
                                          xfail((UnitCubeMesh.create, (2, 2, 2, CellType.Type.hexahedron)))])
def test_evaluate_dofs(mesh_factory):

    func, args = mesh_factory
    mesh = func(*args)

    v = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    w = v*q

    V = FunctionSpace(mesh, v)
    W = FunctionSpace(mesh, w)

    sdim = V.element().space_dimension()
    gdim = V.element().geometric_dimension()

    e = Expression("x[0] + x[1]", degree=1)
    e2 = Expression(["x[0] + x[1]"]*gdim, degree=1)

    coords = numpy.zeros((sdim, gdim), dtype="d")
    coord = numpy.zeros(gdim, dtype="d")
    values0 = numpy.zeros(sdim, dtype="d")
    values1 = numpy.zeros(sdim, dtype="d")
    values2 = numpy.zeros(sdim, dtype="d")
    values3 = numpy.zeros(sdim, dtype="d")
    values4 = numpy.zeros(gdim*sdim, dtype="d")

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    for cell in cells(mesh):
        vx = cell.get_vertex_coordinates()
        orientation = cell.orientation()
        coords = V.element().tabulate_dof_coordinates(cell)
        for i in range(coords.shape[0]):
            coord[:] = coords[i, :]
            values0[i] = e(*coord)
        values1 = L0.element().evaluate_dofs(e, vx, orientation, cell)
        values2 = L01.element().evaluate_dofs(e, vx, orientation, cell)
        values3 = L11.element().evaluate_dofs(e, vx, orientation, cell)
        values4 = L1.element().evaluate_dofs(e2, vx, orientation, cell)

        for i in range(sdim):
            assert round(values0[i] - values1[i], 7) == 0
            assert round(values0[i] - values2[i], 7) == 0
            assert round(values0[i] - values3[i], 7) == 0
            assert round(values4[:sdim][i] - values0[i], 7) == 0
            if gdim == 3:
                assert round(values4[sdim:sdim*2][i] - values0[i], 7) == 0
                assert round(values4[sdim*2:][i] - values0[i], 7) == 0
            else:
                assert round(values4[sdim:][i] - values0[i], 7) == 0


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
        coord = numpy.zeros(gdim, dtype="d")
        values0 = numpy.zeros(sdim, dtype="d")
        values1 = numpy.zeros(sdim, dtype="d")
        for cell in cells(V.mesh()):
            vx = cell.get_vertex_coordinates()
            orientation = cell.orientation()
            coords = V.element().tabulate_dof_coordinates(cell)
            for i in range(coords.shape[0]):
                coord[:] = coords[i, :]
                values0[i] = f(*coord)
            values1 = V.element().evaluate_dofs(f, vx, orientation, cell)
            for i in range(sdim):
                assert round(values0[i] - values1[i], 7) == 0


@pytest.mark.parametrize('mesh_factory', [(UnitSquareMesh, (4, 4)),
                                          (UnitCubeMesh, (2, 2, 2)),
                                          (UnitSquareMesh.create, (4, 4, CellType.Type.quadrilateral)),
                                          (UnitCubeMesh.create, (2, 2, 2, CellType.Type.hexahedron))])
def test_tabulate_coord(mesh_factory):

    func, args = mesh_factory
    mesh = func(*args)

    v = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    w = v*q

    V = FunctionSpace(mesh, v)
    W = FunctionSpace(mesh, w)

    sdim = V.element().space_dimension()
    gdim = V.element().geometric_dimension()
    coord0 = numpy.zeros((sdim, gdim), dtype="d")
    coord1 = numpy.zeros((sdim, gdim), dtype="d")
    coord2 = numpy.zeros((sdim, gdim), dtype="d")
    coord3 = numpy.zeros((sdim, gdim), dtype="d")
    coord4 = numpy.zeros((gdim*sdim, gdim), dtype="d")

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    for cell in cells(mesh):
        coord0 = V.element().tabulate_dof_coordinates(cell)
        coord1 = L0.element().tabulate_dof_coordinates(cell)
        coord2 = L01.element().tabulate_dof_coordinates(cell)
        coord3 = L11.element().tabulate_dof_coordinates(cell)
        coord4 = L1.element().tabulate_dof_coordinates(cell)

        assert (coord0 == coord1).all()
        assert (coord0 == coord2).all()
        assert (coord0 == coord3).all()
        if gdim == 3:
            assert (coord4[sdim:sdim*2] == coord0).all()
            assert (coord4[sdim*2:] == coord0).all()
        else:
            assert (coord4[sdim:] == coord0).all()
