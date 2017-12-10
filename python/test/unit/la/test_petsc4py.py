"""Unit tests for the pybind11 wrapping of PETSc / petsc4py"""

# Copyright (C) 2017 Tormod Landet
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

import gc
from dolfin import (PETScVector, PETScMatrix, UnitSquareMesh, TrialFunction,
                    TestFunction, FunctionSpace, Function, assemble, dx,
                    parameters, as_backend_type)
from dolfin_utils.test import skip_if_not_petsc4py, pushpop_parameters


@skip_if_not_petsc4py
def test_petsc4py_vector(pushpop_parameters):
    "Test PETScVector <-> petsc4py.PETSc.Vec conversions"
    parameters["linear_algebra_backend"] = "PETSc"

    # Assemble a test matrix  
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(V)
    a = v*dx
    b1 = assemble(a)

    # Test conversion dolfin.PETScVector -> petsc4py.PETSc.Vec
    b1 = as_backend_type(b1)
    v1 = b1.vec()

    # Copy and scale vector with petsc4py
    v2 = v1.copy()
    v2.scale(2.0)

    # Test conversion petsc4py.PETSc.Vec  -> PETScVector
    b2 = PETScVector(v2)

    assert (b1.get_local()*2.0 == b2.get_local()).all()


@skip_if_not_petsc4py
def test_petsc4py_matrix(pushpop_parameters):
    "Test PETScMatrix <-> petsc4py.PETSc.Mat conversions"
    parameters["linear_algebra_backend"] = "PETSc"

    # Assemble a test matrix    
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = u*v*dx
    A1 = assemble(a)

    # Test conversion dolfin.PETScMatrix -> petsc4py.PETSc.Mat
    A1 = as_backend_type(A1)
    M1 = A1.mat()

    # Copy and scale matrix with petsc4py
    M2 = M1.copy()
    M2.scale(2.0)

    # Test conversion petsc4py.PETSc.Mat  -> PETScMatrix
    A2 = PETScMatrix(M2)

    assert (A1.array()*2.0 == A2.array()).all()

@skip_if_not_petsc4py
def test_ref_count(pushpop_parameters):
    "Test petsc4py reference counting"
    parameters["linear_algebra_backend"] = "PETSc"

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "P", 1)

    # Check u and x own the vector
    u = Function(V)
    x = as_backend_type(u.vector()).vec()
    assert x.refcount == 2

    # Check decref
    del u; gc.collect()  # destroy u
    assert x.refcount == 1

    # Check incref
    vec = PETScVector(x)
    assert x.refcount == 2
