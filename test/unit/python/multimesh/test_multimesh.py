# -*- coding: utf-8 -*-
"""Unit tests for the MultiMeshFunction class"""

# Copyright (C) 2016 Jørgen S. Dokken
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
# First added:  2016-06-11
# Last changed: 2017-05-28

import pytest
from dolfin import *
import ufl
import numpy

from dolfin_utils.test import fixture, skip_in_parallel

@fixture
def multimesh():
    mesh_0 = RectangleMesh(Point(0,0), Point(1, 1), 20, 20)
    mesh_1 = RectangleMesh(Point(numpy.pi/10,numpy.pi/9),
                           Point(numpy.pi/8,numpy.pi/7),5,8)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()
    return multimesh

@fixture
def V(multimesh):
    element = FiniteElement("Lagrange", triangle, 1)
    return MultiMeshFunctionSpace(multimesh, element)

@fixture
def V_high(multimesh):
    element = FiniteElement("Lagrange", triangle, 3)
    return MultiMeshFunctionSpace(multimesh, element)

@fixture
def f():
    return Expression("sin(pi*x[0])*cos(2*pi*x[1])", degree=4)

@fixture
def f_2():
    return Expression("x[0]*x[1]",degree=2)
@fixture
def v(V):
    return MultiMeshFunction(V)

@fixture
def v_high(V_high):
    return MultiMeshFunction(V_high)

@skip_in_parallel
def test_measure_mul(v, multimesh):
    assert isinstance(v*dX, ufl.form.Form)

@skip_in_parallel
def test_assemble_zero(v, multimesh):
    assert numpy.isclose(assemble_multimesh(v*dX), 0)

@skip_in_parallel
def test_assemble_area(v, multimesh):
    v.vector()[:] = 1
    assert numpy.isclose(assemble_multimesh(v*dX), 1)

@skip_in_parallel
def test_assemble_exterior_facet(v, multimesh):
    v.vector()[:] = 1
    assert numpy.isclose(assemble_multimesh(v*ds), 4)

@skip_in_parallel
def test_interpolate(v_high,f):
    v_high.interpolate(f)
    assert numpy.isclose(assemble_multimesh(v_high*dX), 0)

@skip_in_parallel
def test_project(f,V_high):
    v = project(f,V_high)
    assert numpy.isclose(assemble_multimesh(v*dX), 0)

@skip_in_parallel
def test_errornorm_L2(f_2,v_high):
    const = Expression("1", degree=1)
    v_high.interpolate(f_2)
    assert numpy.isclose(errornorm(const, v_high, norm_type="L2", degree_rise=3), numpy.sqrt(22)/6)

@skip_in_parallel
def test_errornorm_H1(f, f_2, v_high):
    v_high.interpolate(f_2)
    assert numpy.isclose(errornorm(f, v_high, norm_type="H1", degree_rise=3), numpy.sqrt(37./36+5*numpy.pi**2/4))

@fixture
def exactsolution():
    return Expression("x[0]", degree=1)

@pytest.mark.slow
@skip_in_parallel
def test_multimesh_poisson():
    # This tests solves a Poisson problem on two meshes with u = x as
    # exact solution

    # FIXME: This test is quite slow.

    # Solve multimesh Poisson
    mesh_0 = UnitSquareMesh(2, 2)
    mesh_1 = RectangleMesh(Point(0.1*DOLFIN_PI, 0.1*DOLFIN_PI),
                           Point(0.2*DOLFIN_PI, 0.2*DOLFIN_PI),
                           2, 2)

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
    bc = MultiMeshDirichletBC(V, exactsolution(), DomainBoundary())
    bc.apply(A, b)

    # Solve
    uh = MultiMeshFunction(V)
    solve(A, uh.vector(), b)

    # Check error
    assert errornorm(exactsolution(), uh, 'L2', degree_rise=1) < DOLFIN_EPS_LARGE
