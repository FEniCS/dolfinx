"""Unit tests for the DiscreteOperator class"""

# Copyright (C) 2015 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from math import sqrt

import pytest
from petsc4py import PETSc

from dolfin import MPI, FunctionSpace, UnitCubeMesh, UnitSquareMesh
from dolfin.cpp.fem import DiscreteOperators
from dolfin_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_gradient():
    """Test discrete gradient computation (typically used for curl-curl
    AMG preconditioners"""

    def compute_discrete_gradient(mesh):
        V = FunctionSpace(mesh, ("Lagrange", 1))
        W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
        G = DiscreteOperators.build_gradient(W._cpp_object, V._cpp_object)
        assert G.getRefCount() == 1
        num_edges = mesh.num_entities_global(1)
        m, n = G.getSize()
        assert m == num_edges
        assert n == mesh.num_entities_global(0)
        assert round(
            G.norm(PETSc.NormType.FROBENIUS) - sqrt(2.0 * num_edges),
            8) == 0.0

    meshes = [
        UnitSquareMesh(MPI.comm_world, 11, 6),
        UnitCubeMesh(MPI.comm_world, 4, 3, 7)
    ]
    for mesh in meshes:
        compute_discrete_gradient(mesh)


def test_incompatible_spaces():
    "Test that error is thrown when function spaces are not compatible"

    mesh = UnitSquareMesh(MPI.comm_world, 13, 7)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    with pytest.raises(RuntimeError):
        DiscreteOperators.build_gradient(V._cpp_object, W._cpp_object)
    with pytest.raises(RuntimeError):
        DiscreteOperators.build_gradient(V._cpp_object, V._cpp_object)
    with pytest.raises(RuntimeError):
        DiscreteOperators.build_gradient(W._cpp_object, W._cpp_object)

    V = FunctionSpace(mesh, ("Lagrange", 2))
    with pytest.raises(RuntimeError):
        DiscreteOperators.build_gradient(W._cpp_object, V._cpp_object)
