# Copyright (C) 2026 Jack S. Hale, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# The test test_curve_mesh was taken from io4dolfinx and relicensed to LGPLv3
# with permission of Jørgen S. Dokken.

from mpi4py import MPI

import numpy as np
import pytest

from basix import LagrangeVariant
from dolfinx.fem import assemble_scalar, coordinate_element, form, interpolate_geometry
from dolfinx.mesh import (
    CellType,
    create_rectangle,
    create_unit_square,
)
from ufl import ds, dx


def _assert_close_up_to_row_permutation(a, b, atol):
    """Assert rows of a and b are equal, up to a row permutation."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    used = [False] * len(b)
    for ra in a:
        for j, rb in enumerate(b):
            if not used[j] and np.allclose(ra, rb, atol=atol, rtol=0.0):
                used[j] = True
                break
        else:
            raise AssertionError(f"No match for {ra} in {b}")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_geometry_p1_to_p2(dtype):
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype)
    cmap = coordinate_element(CellType.triangle, 2, dtype=dtype)

    new_msh = interpolate_geometry(msh, cmap)
    assert new_msh.geometry.cmap().degree == 2

    # Topology should be the same cpp object.
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmap, new_msh.geometry.dofmap
    assert dm_old.shape[0] == dm_new.shape[0]
    assert dm_new.shape[1] == 6

    atol = 10 * np.finfo(dtype).eps
    for c in range(dm_new.shape[0]):
        v = x_old[dm_old[c]]
        # P2 geometry has original vertices + midpoints.
        expected = np.vstack([v, 0.5 * (v[0] + v[1]), 0.5 * (v[0] + v[2]), 0.5 * (v[1] + v[2])])
        _assert_close_up_to_row_permutation(x_new[dm_new[c]], expected, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_geometry_p1_roundtrip(dtype):
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype)
    cmap = coordinate_element(CellType.triangle, 1, dtype=dtype)

    new_msh = interpolate_geometry(msh, cmap)

    assert new_msh.geometry.cmap().degree == 1
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmap, new_msh.geometry.dofmap
    assert dm_old.shape == dm_new.shape

    atol = 10 * np.finfo(dtype).eps
    for c in range(dm_old.shape[0]):
        np.testing.assert_allclose(x_new[dm_new[c]], x_old[dm_old[c]], atol=atol, rtol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("R", [0.1, 1, 10])
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
def test_curve_mesh(degree, dtype, R, cell_type):
    N = 16
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [[-1.0, -1.0], [1.0, 1.0]],
        [N, N],
        cell_type=cell_type,
        dtype=dtype,
    )
    original_area = form(1 * dx(domain=mesh), dtype=dtype)
    original_circumference = form(1 * ds(domain=mesh), dtype=dtype)

    cmap = coordinate_element(cell_type, degree, variant=LagrangeVariant.equispaced, dtype=dtype)
    curved_mesh = interpolate_geometry(mesh, cmap)

    def transform(x):
        u = R * x[:, 0] * np.sqrt(1.0 - (x[:, 1] ** 2 / (2.0)))
        v = R * x[:, 1] * np.sqrt(1.0 - (x[:, 0] ** 2 / (2.0)))
        return np.column_stack([u, v])

    curved_mesh.geometry.x[:, : curved_mesh.geometry.dim] = transform(curved_mesh.geometry.x)

    area = form(1 * dx(domain=curved_mesh), dtype=dtype)
    circumference = form(1 * ds(domain=curved_mesh), dtype=dtype)

    computed_area = curved_mesh.comm.allreduce(assemble_scalar(area), op=MPI.SUM)
    computed_circumference = curved_mesh.comm.allreduce(assemble_scalar(circumference), op=MPI.SUM)

    tol = 10 * np.finfo(dtype).eps
    # Linear geometry has O(h²) area error, not near machine epsilon.
    if degree > 1:
        assert np.isclose(computed_area, np.pi * R**2, atol=tol)
        assert np.isclose(computed_circumference, 2 * np.pi * R, atol=tol)

    cmap_linear = coordinate_element(cell_type, 1, dtype=dtype)
    linear_mesh = interpolate_geometry(curved_mesh, cmap_linear)
    linear_area = form(1 * dx(domain=linear_mesh), dtype=dtype)
    linear_circumference = form(1 * ds(domain=linear_mesh), dtype=dtype)

    recovered_area = linear_mesh.comm.allreduce(assemble_scalar(linear_area), op=MPI.SUM)
    recovered_circumference = linear_mesh.comm.allreduce(
        assemble_scalar(linear_circumference), op=MPI.SUM
    )

    # Curve original mesh
    mesh.geometry.x[:, : mesh.geometry.dim] = transform(mesh.geometry.x)

    reference_area = mesh.comm.allreduce(assemble_scalar(original_area), op=MPI.SUM)
    assert np.isclose(recovered_area, reference_area, atol=tol)

    reference_circumference = mesh.comm.allreduce(
        assemble_scalar(original_circumference), op=MPI.SUM
    )
    assert np.isclose(recovered_circumference, reference_circumference, atol=tol)
