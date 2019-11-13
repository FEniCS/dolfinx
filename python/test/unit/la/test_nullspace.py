"Unit tests for nullspaces"

# Copyright (C) 2014-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from contextlib import ExitStack

import numpy as np
import pytest

import ufl
from dolfin import (MPI, UnitCubeMesh, UnitSquareMesh, VectorFunctionSpace,
                    cpp, fem, la)
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_matrix
from dolfin.generation import BoxMesh
from ufl import TestFunction, TrialFunction, dx, grad, inner


def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [cpp.la.create_vector(V.dofmap.index_map) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(gdim):
            V.sub(i).dofmap.set(basis[i], 1.0)

        # Build rotational null space basis
        if gdim == 2:
            V.sub(0).set_x(basis[2], -1.0, 1)
            V.sub(1).set_x(basis[2], 1.0, 0)
        elif gdim == 3:
            V.sub(0).set_x(basis[3], -1.0, 1)
            V.sub(1).set_x(basis[3], 1.0, 0)

            V.sub(0).set_x(basis[4], 1.0, 2)
            V.sub(2).set_x(basis[4], -1.0, 0)

            V.sub(2).set_x(basis[5], 1.0, 1)
            V.sub(1).set_x(basis[5], -1.0, 2)

    return la.VectorSpaceBasis(nullspace_basis)


def build_broken_elastic_nullspace(V):
    """Function to build incorrect null space for 2D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [cpp.la.create_vector(V.dofmap.index_map) for i in range(4)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        V.sub(0).dofmap.set(basis[0], 1.0)
        V.sub(1).dofmap.set(basis[1], 1.0)

        # Build rotational null space basis
        V.sub(0).set_x(basis[2], -1.0, 1)
        V.sub(1).set_x(basis[2], 1.0, 0)

        # Add vector that is not in nullspace
        V.sub(1).set_x(basis[3], 1.0, 1)

    return la.VectorSpaceBasis(nullspace_basis)


@pytest.mark.parametrize("mesh", [
    UnitSquareMesh(MPI.comm_world, 12, 13),
    UnitCubeMesh(MPI.comm_world, 12, 18, 15)
])
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_orthogonal(mesh, degree):
    """Test that null spaces orthogonalisation"""
    V = VectorFunctionSpace(mesh, ('Lagrange', degree))
    null_space = build_elastic_nullspace(V)
    assert not null_space.is_orthogonal()
    assert not null_space.is_orthonormal()

    null_space.orthonormalize()
    assert null_space.is_orthogonal()
    assert null_space.is_orthonormal()


@pytest.mark.parametrize("mesh", [
    UnitSquareMesh(MPI.comm_world, 12, 13),
    BoxMesh(
        MPI.comm_world,
        [np.array([0.8, -0.2, 1.2]),
         np.array([3.0, 11.0, -5.0])], [12, 18, 25],
        cell_type=CellType.tetrahedron,
        ghost_mode=GhostMode.none),
])
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_check(mesh, degree):
    V = VectorFunctionSpace(mesh, ('Lagrange', degree))
    u, v = TrialFunction(V), TestFunction(V)

    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)

    E, nu = 2.0e2, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def sigma(w, gdim):
        return 2.0 * mu * ufl.sym(grad(w)) + lmbda * ufl.tr(
            grad(w)) * ufl.Identity(gdim)

    a = inner(sigma(u, mesh.geometry.dim), grad(v)) * dx

    # Assemble matrix and create compatible vector
    A = assemble_matrix(a)
    A.assemble()

    # Create null space basis and test
    null_space = build_elastic_nullspace(V)
    assert null_space.in_nullspace(A, tol=1.0e-8)
    null_space.orthonormalize()
    assert null_space.in_nullspace(A, tol=1.0e-8)

    # Create incorrect null space basis and test
    null_space = build_broken_elastic_nullspace(V)
    assert not null_space.in_nullspace(A, tol=1.0e-8)
    null_space.orthonormalize()
    assert not null_space.in_nullspace(A, tol=1.0e-8)
