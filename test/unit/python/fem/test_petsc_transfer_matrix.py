#!/usr/bin/env py.test

"""Unit tests for the fem interface"""

# Copyright (C) 2016 Chris Richardson
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
import numpy as np
from dolfin import *

from dolfin_utils.test import *

def test_scalar_p1():
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = FunctionSpace(meshc, "CG", 1)
    Vf = FunctionSpace(meshf, "CG", 1)

    u = Expression("x[0] + 2*x[1] + 3*x[2]", degree=1)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = PETScDMCollection.create_transfer_matrix(Vc, Vf)
    Vuc = Function(Vf)
    mat.mult(uc.vector(), Vuc.vector())
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def test_scalar_p2():
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = FunctionSpace(meshc, "CG", 2)
    Vf = FunctionSpace(meshf, "CG", 2)

    u = Expression("x[0]*x[2] + 2*x[1]*x[0] + 3*x[2]", degree=2)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = PETScDMCollection.create_transfer_matrix(Vc, Vf)
    Vuc = Function(Vf)
    mat.mult(uc.vector(), Vuc.vector())
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def test_vector_p1_2d():
    meshc = UnitSquareMesh(3, 3)
    meshf = UnitSquareMesh(5, 5)

    Vc = VectorFunctionSpace(meshc, "CG", 1)
    Vf = VectorFunctionSpace(meshf, "CG", 1)

    u = Expression(("x[0] + 2*x[1]", "4*x[0]"), degree=1)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = PETScDMCollection.create_transfer_matrix(Vc, Vf)
    Vuc = Function(Vf)
    mat.mult(uc.vector(), Vuc.vector())
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def test_vector_p2_2d():
    meshc = UnitSquareMesh(5, 4)
    meshf = UnitSquareMesh(5, 8)

    Vc = VectorFunctionSpace(meshc, "CG", 2)
    Vf = VectorFunctionSpace(meshf, "CG", 2)

    u = Expression(("x[0] + 2*x[1]*x[0]", "4*x[0]*x[1]"), degree=2)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = PETScDMCollection.create_transfer_matrix(Vc, Vf)
    Vuc = Function(Vf)
    mat.mult(uc.vector(), Vuc.vector())
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def test_vector_p1_3d():
    meshc = UnitCubeMesh(2, 3, 4)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = VectorFunctionSpace(meshc, "CG", 1)
    Vf = VectorFunctionSpace(meshf, "CG", 1)

    u = Expression(("x[0] + 2*x[1]", "4*x[0]", "3*x[2] + x[0]"), degree=1)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = PETScDMCollection.create_transfer_matrix(Vc, Vf)
    Vuc = Function(Vf)
    mat.mult(uc.vector(), Vuc.vector())
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def test_taylor_hood_cube():
    pytest.xfail("Problem with Mixed Function Spaces")
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Ve = VectorElement("CG", meshc.ufl_cell(), 2)
    Qe = FiniteElement("CG", meshc.ufl_cell(), 1)
    Ze = MixedElement([Ve, Qe])

    Zc = FunctionSpace(meshc, Ze)
    Zf = FunctionSpace(meshf, Ze)

    z = Expression(("x[0]*x[1]", "x[1]*x[2]", "x[2]*x[0]", "x[0] + 3*x[1] + x[2]"), degree=2)
    zc = interpolate(z, Zc)
    zf = interpolate(z, Zf)

    mat = PETScDMCollection.create_transfer_matrix(Zc, Zf)
    Zuc = Function(Zf)
    mat.mult(zc.vector(), Zuc.vector())
    as_backend_type(Zuc.vector()).update_ghost_values()

    diff = Function(Zf)
    diff.assign(Zuc - zf)
    assert diff.vector().norm("l2") < 1.0e-12
