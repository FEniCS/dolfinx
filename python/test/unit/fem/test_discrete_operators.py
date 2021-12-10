# Copyright (C) 2015 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

import numpy
import pytest

from dolfinx.cpp.fem.petsc import create_discrete_gradient
from dolfinx.fem import FunctionSpace
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import GhostMode
from dolfinx_utils.test.skips import skip_in_parallel

from mpi4py import MPI
from petsc4py import PETSc


@skip_in_parallel
@pytest.mark.parametrize("mesh", [
    UnitSquareMesh(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none),
    UnitSquareMesh(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.shared_facet),
    UnitCubeMesh(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.none),
    UnitCubeMesh(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.shared_facet)
])
def test_gradient(mesh):
    """Test discrete gradient computation (typically used for curl-curl
    AMG preconditioners"""

    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = create_discrete_gradient(W._cpp_object, V._cpp_object)
    assert G.getRefCount() == 1
    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.getSize()
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global
    assert numpy.isclose(G.norm(PETSc.NormType.FROBENIUS), numpy.sqrt(2.0 * num_edges))


def test_incompatible_spaces():
    """Test that error is thrown when function spaces are not compatible"""

    mesh = UnitSquareMesh(MPI.COMM_WORLD, 13, 7)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    with pytest.raises(RuntimeError):
        create_discrete_gradient(V._cpp_object, W._cpp_object)
    with pytest.raises(RuntimeError):
        create_discrete_gradient(V._cpp_object, V._cpp_object)
    with pytest.raises(RuntimeError):
        create_discrete_gradient(W._cpp_object, W._cpp_object)

    V = FunctionSpace(mesh, ("Lagrange", 2))
    with pytest.raises(RuntimeError):
        create_discrete_gradient(W._cpp_object, V._cpp_object)
