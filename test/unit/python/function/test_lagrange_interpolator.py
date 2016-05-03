#!/usr/bin/env py.test

"""Unit tests for interpolation using LagrangeInterpolator"""

# Copyright (C) 2014 Mikael Mortensen
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
# First added:  2014-02-18
# Last changed:

import pytest
import numpy
from dolfin import *


class Quadratic2D(Expression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + 1.0


class Quadratic3D(Expression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + 1.0


def test_functional2D():
    """Test integration of function interpolated in non-matching meshes"""

    f = Quadratic2D(degree=2)

    ll = LagrangeInterpolator()

    # Interpolate quadratic function on course mesh
    mesh0 = UnitSquareMesh(8, 8)
    V0 = FunctionSpace(mesh0, "Lagrange", 2)
    u0 = Function(V0)
    ll.interpolate(u0, f)

    # Interpolate FE function on finer mesh
    mesh1 = UnitSquareMesh(31, 31)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = Function(V1)
    ll.interpolate(u1, u0)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    mesh1 = UnitSquareMesh(30, 30)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = Function(V1)
    ll.interpolate(u1, u0)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0


def test_functional3D():
    """Test integration of function interpolated in non-matching meshes"""

    f = Quadratic3D(degree=2)

    ll = LagrangeInterpolator()

    # Interpolate quadratic function on course mesh
    mesh0 = UnitCubeMesh(4, 4, 4)
    V0 = FunctionSpace(mesh0, "Lagrange", 2)
    u0 = Function(V0)
    ll.interpolate(u0, f)

    # Interpolate FE function on finer mesh
    mesh1 = UnitCubeMesh(11, 11, 11)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = Function(V1)
    ll.interpolate(u1, u0)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    mesh1 = UnitCubeMesh(10, 11, 10)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = Function(V1)
    ll.interpolate(u1, u0)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0
