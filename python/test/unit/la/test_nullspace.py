# Copyright (C) 2014-2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for nullspaces"""

from contextlib import ExitStack

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from dolfinx import la
from dolfinx.fem import form, functionspace
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.la import create_petsc_vector
from dolfinx.mesh import CellType, GhostMode, create_box, create_unit_cube, create_unit_square
from ufl import TestFunction, TrialFunction, dx, grad, inner


def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    ns = [la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.flatten() for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][dofs[i]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.flatten()
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        if gdim == 2:
            basis[2][dofs[0]] = -x1
            basis[2][dofs[1]] = x0
        elif gdim == 3:
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0
            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2

    return ns


def build_broken_elastic_nullspace(V):
    """Function to build incorrect null space for 2D elasticity"""

    # Create list of vectors for null space
    ns = [create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs) for i in range(4)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.flatten() for i in range(2)]
        basis[0][dofs[0]] = 1.0
        basis[1][dofs[1]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.flatten()
        x0, x1 = x[dofs_block, 0], x[dofs_block, 1]
        basis[2][dofs[0]] = -x1
        basis[2][dofs[1]] = x0

        # Add vector that is not in nullspace
        basis[3][dofs[1]] = x1

    return ns


@pytest.mark.parametrize(
    "mesh",
    [create_unit_square(MPI.COMM_WORLD, 12, 13), create_unit_cube(MPI.COMM_WORLD, 12, 18, 15)],
)
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_orthogonal(mesh, degree):
    """Test that null spaces orthogonalisation"""
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", degree, (gdim,)))
    nullspace = build_elastic_nullspace(V)
    assert not la.is_orthonormal(nullspace, eps=1.0e-4)
    la.orthonormalize(nullspace)
    assert la.is_orthonormal(nullspace, eps=1.0e-3)
    for x in nullspace:
        x.destroy()


@pytest.mark.parametrize(
    "mesh",
    [
        create_unit_square(MPI.COMM_WORLD, 12, 13),
        create_box(
            MPI.COMM_WORLD,
            [np.array([0.8, -0.2, 1.2]), np.array([3.0, 11.0, -5.0])],
            [12, 18, 25],
            cell_type=CellType.tetrahedron,
            ghost_mode=GhostMode.none,
        ),
    ],
)
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_check(mesh, degree):
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", degree, (gdim,)))
    u, v = TrialFunction(V), TestFunction(V)

    E, nu = 2.0e2, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def sigma(w, gdim):
        return 2.0 * mu * ufl.sym(grad(w)) + lmbda * ufl.tr(grad(w)) * ufl.Identity(gdim)

    a = form(inner(sigma(u, mesh.geometry.dim), grad(v)) * dx)

    # Assemble matrix and create compatible vector
    A = assemble_matrix(a)
    A.assemble()

    # Create null space basis and test
    nullspace = build_elastic_nullspace(V)
    la.orthonormalize(nullspace)
    ns = PETSc.NullSpace().create(vectors=nullspace)
    assert ns.test(A)

    # Create incorrect null space basis and test
    nullspace = build_broken_elastic_nullspace(V)
    la.orthonormalize(nullspace)
    ns = PETSc.NullSpace().create(vectors=nullspace)
    assert not ns.test(A)
    for x in nullspace:
        x.destroy()
