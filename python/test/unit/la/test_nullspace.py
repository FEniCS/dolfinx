# Copyright (C) 2014-2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for nullspaces."""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import la
from dolfinx.fem import assemble_matrix, form, functionspace
from dolfinx.mesh import CellType, GhostMode, create_box, create_unit_cube, create_unit_square
from ufl import TestFunction, TrialFunction, dx, grad, inner


def build_elastic_nullspace(V, dtype):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    ns = [la.vector(V.dofmap.index_map, V.dofmap.index_map_bs) for i in range(dim)]

    basis = [x.array for x in ns]
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_orthogonal(gdim, degree, dtype):
    """Test null spaces orthogonalisation"""
    xtype = dtype(0).real.dtype
    if gdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 12, 13, dtype=xtype)
    elif gdim == 3:
        mesh = create_unit_cube(MPI.COMM_WORLD, 12, 18, 15, dtype=xtype)

    V = functionspace(mesh, ("Lagrange", degree, (gdim,)))
    nullspace = build_elastic_nullspace(V, dtype)
    assert not la.is_orthonormal(nullspace, eps=1.0e-4)
    la.orthonormalize(nullspace)
    assert la.is_orthonormal(nullspace, eps=1.0e-3)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_nullspace_check(gdim, degree, dtype):
    """Test that elasticity nullspace is actually a nullspace."""

    # TODO: Once we support SpMV, run on MPI.COMM_WORLD
    comm = MPI.COMM_SELF
    xtype = dtype(0).real.dtype
    if gdim == 2:
        mesh = create_unit_square(comm, 12, 13, dtype=xtype)
    elif gdim == 3:
        mesh = create_box(
            comm,
            [np.array([0.8, -0.2, 1.2]), np.array([3.0, 11.0, -5.0])],
            [12, 18, 25],
            cell_type=CellType.tetrahedron,
            ghost_mode=GhostMode.none,
            dtype=xtype,
        )

    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", degree, (gdim,)))
    u, v = TrialFunction(V), TestFunction(V)

    E, nu = 2.0e2, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def sigma(w, gdim):
        return 2.0 * mu * ufl.sym(grad(w)) + lmbda * ufl.tr(grad(w)) * ufl.Identity(gdim)

    a = form(inner(sigma(u, mesh.geometry.dim), grad(v)) * dx, dtype=dtype)

    # Assemble matrix and create compatible vector
    A = assemble_matrix(a)
    A.scatter_reverse()

    # Create null space basis and test
    nullspace = build_elastic_nullspace(V, dtype)
    la.orthonormalize(nullspace)
    As = A.to_scipy()

    eps = np.sqrt(np.finfo(dtype).eps)
    for x in nullspace:
        assert np.isclose(np.linalg.norm(As * nullspace[0].array), 0, atol=eps)
