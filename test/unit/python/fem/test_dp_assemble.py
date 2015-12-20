#!/usr/bin/env py.test

"""Unit tests for dP assembly"""

# Copyright (C) 2014 Johan Hake
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

import pytest
import os
import numpy as np
from dolfin import *

from dolfin_utils.test import *


@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param


def _create_dp_problem(dim):
    assert dim in [1, 2, 3]
    if dim == 1:
        mesh = UnitIntervalMesh(20)
    elif dim == 2:
        mesh = UnitSquareMesh(10, 10)
    else:
        mesh = UnitCubeMesh(4, 4, 4)

    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)
    VV = FunctionSpace(mesh, P1*P1)

    # Create expression used to interpolate initial data
    y_dim = 1 if dim > 1 else 0
    z_dim = 2 if dim > 2 else (1 if dim > 1 else 0)
    e = Expression("x[0]+2*x[{}]+x[{}]".format(y_dim, z_dim))
    ee = Expression(["x[0]+x[{}]".format(z_dim),
                     "x[0]*x[{}]+x[{}]".format(y_dim, z_dim)])

    # Create coefficients
    u = interpolate(e, V)
    uu = interpolate(ee, VV)

    # Test and trial spaces
    v = TestFunction(V)
    vv = TestFunction(VV)
    U = TrialFunction(V)
    UU = TrialFunction(VV)

    # Subdomains
    subdomain = AutoSubDomain(lambda x, on_boundary: x[0] <= 0.5)
    disjoint_subdomain = AutoSubDomain(lambda x, on_boundary: x[0] > 0.5)
    vertex_domain = VertexFunction("size_t", mesh, 0)
    subdomain.mark(vertex_domain, 1)
    bc = DirichletBC(VV, Constant((0, 0)), disjoint_subdomain)
    dPP = dP(subdomain_data=vertex_domain)

    return (u, uu), (v, vv), (U, UU), dPP, bc


@use_gc_barrier
def test_scalar_assemble(dim):
    eps = 1000*DOLFIN_EPS

    (u, uu), (v, vv), (U, UU), dPP, bc = _create_dp_problem(dim)

    scalar_value = assemble(u*dP)
    assert abs(scalar_value-u.vector().sum()) < eps

    scalar_value = assemble((uu[0]+uu[1])*dPP)
    assert abs(scalar_value-uu.vector().sum()) < eps

    scalar_value = assemble((uu[0]+uu[1])*dPP(1))
    bc.apply(uu.vector())
    assert abs(scalar_value-uu.vector().sum()) < eps


@use_gc_barrier
def test_vector_assemble(dim):
    eps = 1000*DOLFIN_EPS

    (u, uu), (v, vv), (U, UU), dPP, bc = _create_dp_problem(dim)

    # In parallel vec.array() will return only local to process values
    vec = assemble(u*v*dPP)
    assert sum(np.absolute(vec.array() - u.vector().array())) < eps

    vec = assemble(inner(uu, vv)*dP)
    assert sum(np.absolute(vec.array() - uu.vector().array())) < eps

    vec = assemble(inner(uu, vv)*dPP(1))
    bc.apply(uu.vector())
    assert sum(np.absolute(vec.array() - uu.vector().array())) < eps


@use_gc_barrier
def test_matrix_assemble(dim):
    eps = 1000*DOLFIN_EPS

    (u, uu), (v, vv), (U, UU), dPP, bc = _create_dp_problem(dim)

    # Scalar assemble
    mat = assemble(u*v*U*dPP)

    # Create a numpy matrix based on the local size of the vector
    # and populate it with values from local vector
    loc_range = u.vector().local_range()
    vec_mat = np.zeros_like(mat.array())
    vec_mat[range(loc_range[1] - loc_range[0]),
            range(loc_range[0], loc_range[1])] = u.vector().array()

    assert np.sum(np.absolute(mat.array() - vec_mat)) < eps

    # Vector assemble
    mat = assemble((uu[0]*vv[0]*UU[0] + uu[1]*vv[1]*UU[1])*dPP)

    # Create a numpy matrix based on the local size of the vector
    # and populate it with values from local vector
    loc_range = uu.vector().local_range()
    vec_mat = np.zeros_like(mat.array())
    vec_mat[range(loc_range[1] - loc_range[0]),
            range(loc_range[0], loc_range[1])] = uu.vector().array()

    assert np.sum(np.absolute(mat.array() - vec_mat)) < eps
