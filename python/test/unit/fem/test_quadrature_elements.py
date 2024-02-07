# Copyright (C) 2023 Jorgen Dokken and Matthew Scroggs
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
def test_default(degree):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    CG2_vect = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    Qe = basix.ufl.quadrature_element(msh.topology.cell_name(), degree=degree)
    Quad = dolfinx.fem.functionspace(msh, Qe)

    u = dolfinx.fem.Function(Quad)
    v = ufl.TrialFunction(CG2_vect)

    dx_m = ufl.Measure(
        "dx", domain=msh, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"}
    )
    ds = ufl.Measure("ds", domain=msh)

    residual = u * v * dx_m
    vol = dolfinx.fem.form(residual)
    residual = v * ds
    surf = dolfinx.fem.form(residual)

    residual = u * v * dx_m + v * ds
    vol_surf = dolfinx.fem.form(residual)

    vol_v = dolfinx.fem.assemble_vector(vol)
    sur_v = dolfinx.fem.assemble_vector(surf)

    vol_surf = dolfinx.fem.assemble_vector(vol_surf)

    assert np.allclose(vol_v.array + sur_v.array, vol_surf.array)


def test_points_and_weights():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    CG2_vect = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    Qe = basix.ufl.quadrature_element(
        msh.topology.cell_name(),
        value_shape=(),
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1 / 3, 1 / 3]]),
        weights=np.array([0.2, 0.2, 0.2, 0.4]),
    )
    Quad = dolfinx.fem.functionspace(msh, Qe)

    u = dolfinx.fem.Function(Quad)
    v = ufl.TrialFunction(CG2_vect)

    dx_m = ufl.Measure(
        "dx", domain=msh, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"}
    )
    ds = ufl.Measure("ds", domain=msh)

    residual = u * v * dx_m
    vol = dolfinx.fem.form(residual)
    residual = v * ds
    surf = dolfinx.fem.form(residual)

    residual = u * v * dx_m + v * ds
    vol_surf = dolfinx.fem.form(residual)

    vol_v = dolfinx.fem.assemble_vector(vol)
    sur_v = dolfinx.fem.assemble_vector(surf)

    vol_surf = dolfinx.fem.assemble_vector(vol_surf)

    assert np.allclose(vol_v.array + sur_v.array, vol_surf.array)


@pytest.mark.parametrize("degree", range(1, 5))
def test_interpolation(degree):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    e = basix.ufl.quadrature_element(msh.topology.cell_name(), degree=degree)
    space = dolfinx.fem.functionspace(msh, e)
    p4 = dolfinx.fem.functionspace(msh, ("Lagrange", 4))

    f_p4 = dolfinx.fem.Function(p4)
    f_p4.interpolate(lambda x: x[0] ** 4)

    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: x[0] ** 4)

    diff = dolfinx.fem.form(ufl.inner(f - f_p4, f - f_p4) * ufl.dx)

    error = dolfinx.fem.assemble_scalar(diff)

    assert np.isclose(error, 0)


@pytest.mark.parametrize("degree", range(1, 5))
def test_interpolation_blocked(degree):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    e = basix.ufl.quadrature_element(msh.topology.cell_name(), value_shape=(2,), degree=degree)
    space = dolfinx.fem.functionspace(msh, e)
    p4 = dolfinx.fem.functionspace(msh, ("Lagrange", 4, (2,)))

    f_p4 = dolfinx.fem.Function(p4)
    f_p4.interpolate(lambda x: ([x[1] ** 4, x[0] ** 3]))

    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: ([x[1] ** 4, x[0] ** 3]))

    diff = dolfinx.fem.form(ufl.inner(f - f_p4, f - f_p4) * ufl.dx)

    error = dolfinx.fem.assemble_scalar(diff)

    assert np.isclose(error, 0)


def extract_diagonal(mat):
    bs = mat.block_size[0]
    diag = np.empty(len(mat.indices) * bs)
    for row, (start, end) in enumerate(zip(mat.indptr[:-1], mat.indptr[1:])):
        for i in range(start, end):
            if mat.indices[i] == row:
                for block in range(bs):
                    diag[bs * row + block] = mat.data[bs**2 * i + (bs + 1) * block]
    return diag


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("shape", [(), (1,), (2,), (3,), (4,), (2, 2), (3, 3)])
def test_vector_element(shape):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    dx_m = ufl.Measure(
        "dx",
        domain=msh,
        metadata={"quadrature_degree": 1, "quadrature_scheme": "default"},
    )

    Qe = basix.ufl.quadrature_element(
        msh.topology.cell_name(), value_shape=shape, scheme="default", degree=1
    )
    Quad = dolfinx.fem.functionspace(msh, Qe)
    q_ = ufl.TestFunction(Quad)
    dq = ufl.TrialFunction(Quad)
    one = dolfinx.fem.Function(Quad)
    one.x.array[:] = 1.0
    mass_L_form = dolfinx.fem.form(ufl.inner(one, q_) * dx_m)
    mass_v = dolfinx.fem.assemble_vector(mass_L_form)
    mass_a_form = dolfinx.fem.form(ufl.inner(dq, q_) * dx_m)
    mass_A = dolfinx.fem.assemble_matrix(mass_a_form)

    assert np.allclose(extract_diagonal(mass_A), mass_v.array)
