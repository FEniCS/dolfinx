#!/usr/bin/env py.test

"""Unit tests for the KrylovSolver interface"""

# Copyright (C) 2016 INSERT NAME HERE
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
from dolfin import *
from dolfin_utils.test import skip_if_not_PETsc_or_not_slepc


@skip_if_not_PETsc_or_not_slepc
def test_slepc_eigensolver_gen_hermitian():
    "Test SLEPc eigen solver"

    # Set backend
    parameters["linear_algebra_backend"] = "PETSc"

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)

    u, v = TrialFunction(V), TestFunction(V)
    k = dot(grad(u), grad(v))*dx
    m = u*v*dx
    L = Constant(1.0)*v*dx

    K = PETScMatrix()
    M = PETScMatrix()
    x0 = PETScVector()

    assemble_system(k, L, bcs=[], A_tensor=K, b_tensor=x0)
    assemble_system(m, L, bcs=[], A_tensor=M, b_tensor=x0)

    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "krylov-schur"
    esolver.parameters["spectral_transform"] = 'shift-and-invert'
    esolver.parameters['spectral_shift'] = 0.0
    esolver.parameters["problem_type"] = 'gen_hermitian'

    nevs = 20
    esolver.solve(20)

    for j in range(1, nevs):
      re, im = esolver.get_eigenvalue(j)
      assert re > 0.0
      assert near(im, 0.0)


@skip_if_not_PETsc_or_not_slepc
def test_slepc_null_space():
    "Test SLEPc eigen solver"

    # Set backend
    parameters["linear_algebra_backend"] = "PETSc"

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)

    u, v = TrialFunction(V), TestFunction(V)
    k = dot(grad(u), grad(v))*dx
    m = u*v*dx
    L = Constant(1.0)*v*dx

    K = PETScMatrix()
    M = PETScMatrix()
    x0 = PETScVector()

    assemble_system(k, L, bcs=[], A_tensor=K, b_tensor=x0)
    assemble_system(m, L, bcs=[], A_tensor=M, b_tensor=x0)

    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = 'gen_hermitian'
    
    u0 = Function(V)
    nullspace_basis = as_backend_type(u0.vector().copy())
    V.dofmap().set(nullspace_basis, 1.0)
    esolver.set_deflation_space(nullspace_basis)

    nevs = 20
    esolver.solve(20)

    for j in range(1, nevs):
      re, im = esolver.get_eigenvalue(j)
      assert re > 0.0
      assert near(im, 0.0)

@skip_if_not_PETsc_or_not_slepc
def test_slepc_vector_null_space():

    def build_nullspace(V, x):
        nullspace_basis = [x.copy() for i in range(2)]

        V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
        V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

        for x in nullspace_basis:
            x.apply("insert")

        # Create vector space basis and orthogonalize
        basis = VectorSpaceBasis(nullspace_basis)
        basis.orthonormalize()

        return basis

    # Set backend
    parameters["linear_algebra_backend"] = "PETSc"

    mesh = UnitSquareMesh(8, 8)
    V = VectorFunctionSpace(mesh, "CG", 1)

    u, v = TrialFunction(V), TestFunction(V)
    k = inner(grad(u), grad(v))*dx
    m = dot(u, v)*dx
    L = dot(Constant((0.0, 0.0)), v)*dx

    K = PETScMatrix()
    M = PETScMatrix()
    x0 = PETScVector()

    assemble_system(k, L, bcs=[], A_tensor=K, b_tensor=x0)
    assemble_system(m, L, bcs=[], A_tensor=M, b_tensor=x0)

    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = 'gen_hermitian'
    
    u0 = Function(V)
    nullspace_basis = build_nullspace(V, u0.vector())
    esolver.set_deflation_space(nullspace_basis)

    nevs = 20
    esolver.solve(20)

    for j in range(1, nevs):
      re, im = esolver.get_eigenvalue(j)
      assert re > 0.0
      assert near(im, 0.0)

