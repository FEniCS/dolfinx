"""Unit tests for coordinates interface"""

# Copyright (C) 2016 Jan Blechta
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

from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import FunctionSpace, VectorFunctionSpace, Function, MPI
from dolfin import UserExpression
from dolfin import get_coordinates, set_coordinates, Mesh
from dolfin import Expression, interpolate
from dolfin_utils.test import skip_in_parallel, fixture


@fixture
def meshes_p1():
    return UnitIntervalMesh(MPI.comm_world, 10), UnitSquareMesh(MPI.comm_world, 3, 3), UnitCubeMesh(MPI.comm_world, 2, 2, 2)


def _test_get_set_coordinates(mesh):
    # Get coords
    V = FunctionSpace(mesh, mesh.ufl_coordinate_element())
    c = Function(V)
    get_coordinates(c, mesh.geometry())

    # Check correctness of got coords
    _check_coords(mesh, c)

    # Backup and zero coords
    coords = mesh.geometry().x()
    coords_old = coords.copy()
    coords[:] = 0.0
    assert np.all(mesh.geometry().x() == 0.0)

    # Set again to old value
    set_coordinates(mesh.geometry(), c)

    # Check
    assert np.all(mesh.geometry().x() == coords_old)

def _check_coords(mesh, c):
    # FIXME: This does not work for higher-order geometries although it should
    if mesh.geometry().degree() > 1:
        return

    # Compare supplied c with interpolation of x
    class X(UserExpression):
        def eval(self, values, x):
            values[:] = x[:]
    x = X(domain=mesh, element=mesh.ufl_coordinate_element())
    x = interpolate(x, c.function_space())
    vec = x.vector()
    vec -= c.vector()
    assert np.isclose(vec.norm("l1"), 0.0)


def test_linear(meshes_p1):
    for mesh in meshes_p1:
        _test_get_set_coordinates(mesh)


def test_raises(meshes_p1):
    mesh1, mesh2 = meshes_p1[:2]

    # Wrong FE family
    V = VectorFunctionSpace(mesh2, "Discontinuous Lagrange", 1)
    c = Function(V)
    with pytest.raises(RuntimeError):
        get_coordinates(c, mesh2.geometry())
    with pytest.raises(RuntimeError):
        set_coordinates(mesh2.geometry(), c)

    # Wrong value rank
    V = FunctionSpace(mesh2, "Lagrange", 1)
    c = Function(V)
    with pytest.raises(RuntimeError):
        get_coordinates(c, mesh2.geometry())
    with pytest.raises(RuntimeError):
        set_coordinates(mesh2.geometry(), c)

    # Wrong value shape
    V = VectorFunctionSpace(mesh2, "Lagrange", mesh2.geometry().degree(),
            dim=mesh2.geometry().dim() - 1)
    c = Function(V)
    with pytest.raises(RuntimeError):
        get_coordinates(c, mesh2.geometry())
    with pytest.raises(RuntimeError):
        set_coordinates(mesh2.geometry(), c)

    # Non-matching degree
    V = VectorFunctionSpace(mesh2, "Lagrange", mesh2.geometry().degree() + 1)
    c = Function(V)
    with pytest.raises(RuntimeError):
        get_coordinates(c, mesh2.geometry())
    with pytest.raises(RuntimeError):
        set_coordinates(mesh2.geometry(), c)
