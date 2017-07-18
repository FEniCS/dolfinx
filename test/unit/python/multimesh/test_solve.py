"""Unit tests for MultiMesh PDE solvers"""

# Copyright (C) 2017 August Johansson
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
#
# First added:  2017-07-18
# Last changed: 2017-07-18

import pytest
from dolfin import *

from dolfin_utils.test import skip_in_parallel, fixture


@fixture
def exactsolution_2d():
    return Expression("x[0] + x[1]", degree=1)

@fixture
def exactsolution_3d():
    return Expression("x[0] + x[1] + x[2]", degree=1)

@fixture
def solve_multimesh_poisson(mesh_0, mesh_1, exactsolution):

    # Build multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()

    # FEM setup
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set data
    f = Constant(0.0)

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form
    a = dot(grad(u), grad(v))*dX \
      - dot(avg(grad(u)), jump(v, n))*dI \
      - dot(avg(grad(v)), jump(u, n))*dI \
      + alpha/h*jump(u)*jump(v)*dI \
      + beta*dot(jump(grad(u)), jump(grad(v)))*dO

    # Define linear form
    L = f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    # Apply boundary condition
    bc = MultiMeshDirichletBC(V, exactsolution, DomainBoundary())
    bc.apply(A, b)

    # Solve
    uh = MultiMeshFunction(V)
    solve(A, uh.vector(), b)

    return uh

    
@pytest.mark.slow
@skip_in_parallel
def test_multimesh_poisson_2d():
    # This tests solves a Poisson problem on two meshes in 2D with u =
    # x+y as exact solution

    # FIXME: This test is quite slow.

    # Define meshes
    mesh_0 = UnitSquareMesh(2, 2)
    mesh_1 = RectangleMesh(Point(0.1*DOLFIN_PI, 0.1*DOLFIN_PI),
                           Point(0.2*DOLFIN_PI, 0.2*DOLFIN_PI),
                           2, 2)

    # Solve multimesh Poisson
    uh = solve_multimesh_poisson(mesh_0, mesh_1, exactsolution_2d())
    
    # Check error
    assert errornorm(exactsolution_2d(), uh, 'L2', degree_rise=1) < DOLFIN_EPS_LARGE

@pytest.mark.slow
@skip_in_parallel
@pytest.mark.skipif(True, reason="3D not fully implemented")
def test_multimesh_poisson_3d():
    # This tests solves a Poisson problem on two meshes in 3D with u =
    # x+y+z as exact solution

    # FIXME: This test is quite slow.

    # Define meshes
    mesh_0 = UnitCubeMesh(2, 2, 2)
    mesh_1 = BoxMesh(Point(0.1*DOLFIN_PI, 0.1*DOLFIN_PI, 0.1*DOLFIN_PI),
                     Point(0.2*DOLFIN_PI, 0.2*DOLFIN_PI, 0.2*DOLFIN_PI),
                     2, 2, 2)
    
    # Solve multimesh Poisson
    uh = solve_multimesh_poisson(mesh_0, mesh_1, exactsolution_3d())
    
    # Check error
    assert errornorm(exactsolution_3d(), uh, 'L2', degree_rise=1) < DOLFIN_EPS_LARGE
