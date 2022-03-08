# Copyright (C) 2015 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

import numpy as np
import pytest

import ufl
from dolfinx.cpp.fem.petsc import create_discrete_gradient
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.mesh import GhostMode, create_unit_cube, create_unit_square

from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("mesh", [
    create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none),
    create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.shared_facet),
    create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.none),
    create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.shared_facet)
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
    assert np.isclose(G.norm(PETSc.NormType.FROBENIUS), np.sqrt(2.0 * num_edges))


def test_incompatible_spaces():
    """Test that error is thrown when function spaces are not compatible"""

    mesh = create_unit_square(MPI.COMM_WORLD, 13, 7)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    with pytest.raises(RuntimeError):
        create_discrete_gradient(V._cpp_object, W._cpp_object)
    with pytest.raises(RuntimeError):
        create_discrete_gradient(V._cpp_object, V._cpp_object)
    with pytest.raises(RuntimeError):
        create_discrete_gradient(W._cpp_object, W._cpp_object)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("mesh", [
    create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none),
    create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, ghost_mode=GhostMode.none),
])
@pytest.mark.parametrize("p", range(1, 4))
def test_interpolation_matrix(mesh, p):
    V = FunctionSpace(mesh, ("Lagrange", p))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", p))
    G = create_discrete_gradient(W._cpp_object, V._cpp_object)

    u = Function(V)
    u.interpolate(lambda x: 2 * x[0]**p + 3 * x[1]**p)
    grad_u = Expression(ufl.grad(u), W.element.interpolation_points)
    w = Function(W)
    w.interpolate(grad_u)
    w.x.scatter_forward()

    # Compute global matrix vector product
    w_2 = G[:, :] @ u.x.array
    w2 = Function(W)
    w2.x.array[:] = w_2
    w2.name = "W_new"

    w.name = "W_old"
    import dolfinx.io
    with dolfinx.io.XDMFFile(mesh.comm, "u.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(w2)
        xdmf.write_function(w)

    assert np.allclose(w.x.array, w_2)
