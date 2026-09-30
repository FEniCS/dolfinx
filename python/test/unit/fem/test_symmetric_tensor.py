# Copyright (C) 2023-2026 Matthew W. Scroggs and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("symmetry", [True, False])
def test_transpose(degree, symmetry):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    e = basix.ufl.element(
        "Lagrange",
        "triangle",
        degree,
        shape=(2, 2),
        symmetry=symmetry,
        dtype=dolfinx.default_real_type,
    )
    space = dolfinx.fem.functionspace(mesh, e)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: [x[0], x[1], 2 * x[1], x[0] ** 3])

    form = dolfinx.fem.form(ufl.inner(f - ufl.transpose(f), f - ufl.transpose(f)) * ufl.dx)
    assert np.isclose(dolfinx.fem.assemble_scalar(form), 0) == symmetry


def test_interpolation():
    """Test that a symmetric 3x3 2-tensor is correctly interpolated."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def tensor(x):
        mat = np.array([[0], [1], [2], [1], [3], [4], [2], [4], [5]])
        return np.broadcast_to(mat, (9, x.shape[1]))

    element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=False, dtype=dolfinx.default_real_type
    )
    space = dolfinx.fem.functionspace(mesh, element)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: tensor(x))

    symm_element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True, dtype=dolfinx.default_real_type
    )
    symm_space = dolfinx.fem.functionspace(mesh, symm_element)
    symm_f = dolfinx.fem.Function(symm_space)
    symm_f.interpolate(lambda x: tensor(x))

    l2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form((f - symm_f) ** 2 * ufl.dx))
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.isclose(l2_error, 0.0, atol=atol)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("symmetry", [True, False])
def test_interpolation_tensor_to_sym(dim, symmetry):
    """Test interpolation from a Regge function into a (symmetric) 2-tensor space."""
    comm = MPI.COMM_WORLD
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(comm, 5, 5)
    else:
        mesh = dolfinx.mesh.create_unit_cube(comm, 5, 5, 5)

    def tensor(x):
        # symmetric and linear tensor: a_ij = a_i + a_j + δ_ij
        points = x[:dim, None]
        A = points + points.swapaxes(0, 1) + np.eye(dim)[:, :, None]
        return A.reshape(dim * dim, -1)

    regge_element = basix.ufl.element(
        "Regge", mesh.basix_cell(), 1, dtype=dolfinx.default_real_type
    )
    u_regge = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, regge_element))
    u_regge.interpolate(tensor)

    element = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        1,
        shape=(dim, dim),
        symmetry=symmetry,
        dtype=dolfinx.default_real_type,
    )
    u_lagrange = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element))
    u_lagrange.interpolate(u_regge)

    l2_error = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form((u_lagrange - u_regge) ** 2 * ufl.dx))
    )
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.isclose(l2_error, 0.0, atol=atol)


def test_eval():
    """Test that eval is correct for a symmetric 3x3 2-tensor is correct."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    mat = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5])

    def tensor(x):
        return np.broadcast_to(mat.reshape((9, 1)), (9, x.shape[1]))

    element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True, dtype=dolfinx.default_real_type
    )
    space = dolfinx.fem.functionspace(mesh, element)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: tensor(x))
    value = f.eval([[0, 0, 0]], [0])
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.allclose(value, mat, atol=atol)
