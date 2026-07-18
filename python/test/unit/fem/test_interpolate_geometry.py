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
    assert new_msh.geometry.cmaps[0].degree == 2

    # Topology should be the same cpp object.
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmaps[0], new_msh.geometry.dofmaps[0]
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

    assert new_msh.geometry.cmaps[0].degree == 1
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmaps[0], new_msh.geometry.dofmaps[0]
    assert dm_old.shape == dm_new.shape

    atol = 10 * np.finfo(dtype).eps
    for c in range(dm_old.shape[0]):
        np.testing.assert_allclose(x_new[dm_new[c]], x_old[dm_old[c]], atol=atol, rtol=0.0)


def _curve_mesh_errors(N, degree, dtype, R, cell_type, lagrange_variant):
    """Return (area_error, circumference_error) for a degree-p curved disk mesh with N cells."""
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [[-1.0, -1.0], [1.0, 1.0]],
        [N, N],
        cell_type=cell_type,
        dtype=dtype,
    )
    original_area_form = form(1 * dx(domain=mesh), dtype=dtype)
    original_circ_form = form(1 * ds(domain=mesh), dtype=dtype)

    def transform(x):
        x_c = np.zeros_like(x)
        x_c[:, 0] = R * x[:, 0] * np.sqrt(1.0 - (x[:, 1] ** 2 / (2.0)))
        x_c[:, 1] = R * x[:, 1] * np.sqrt(1.0 - (x[:, 0] ** 2 / (2.0)))
        return x_c

    cmap = coordinate_element(cell_type, degree, variant=lagrange_variant, dtype=dtype)
    curved_mesh = interpolate_geometry(mesh, cmap)
    curved_mesh.geometry.x[:] = transform(curved_mesh.geometry.x)

    area_form = form(1 * dx(domain=curved_mesh), dtype=dtype)
    circ_form = form(1 * ds(domain=curved_mesh), dtype=dtype)

    area = curved_mesh.comm.allreduce(assemble_scalar(area_form), op=MPI.SUM)
    circ = curved_mesh.comm.allreduce(assemble_scalar(circ_form), op=MPI.SUM)

    # Interpolate curved mesh back to P1 and compare against the original P1 mesh
    # with the same transform applied — should match to near machine precision.
    cmap_linear = coordinate_element(cell_type, 1, dtype=dtype)
    linear_mesh = interpolate_geometry(curved_mesh, cmap_linear)
    linear_area_form = form(1 * dx(domain=linear_mesh), dtype=dtype)
    linear_circ_form = form(1 * ds(domain=linear_mesh), dtype=dtype)

    recovered_area = linear_mesh.comm.allreduce(assemble_scalar(linear_area_form), op=MPI.SUM)
    recovered_circ = linear_mesh.comm.allreduce(assemble_scalar(linear_circ_form), op=MPI.SUM)

    mesh.geometry.x[:] = transform(mesh.geometry.x)
    reference_area = mesh.comm.allreduce(assemble_scalar(original_area_form), op=MPI.SUM)
    reference_circ = mesh.comm.allreduce(assemble_scalar(original_circ_form), op=MPI.SUM)

    tol = 100 * np.finfo(dtype).eps
    assert np.isclose(recovered_area, reference_area, rtol=tol)
    assert np.isclose(recovered_circ, reference_circ, rtol=tol)

    return abs(area - np.pi * R**2), abs(circ - 2.0 * np.pi * R)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("R", [0.1])
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize(
    "lagrange_variant", [LagrangeVariant.equispaced, LagrangeVariant.gll_isaac]
)
def test_curve_mesh(degree, dtype, R, cell_type, lagrange_variant):
    Ns = [4, 8, 16, 32]
    errors = [_curve_mesh_errors(N, degree, dtype, R, cell_type, lagrange_variant) for N in Ns]

    area_errors = np.array([e[0] for e in errors])
    circ_errors = np.array([e[1] for e in errors])

    hs = np.array([2.0 / N for N in Ns])
    area_rate = np.polyfit(np.log(hs), np.log(area_errors), 1)[0]
    circ_rate = np.polyfit(np.log(hs), np.log(circ_errors), 1)[0]

    # np.float32 converges to machine epsilon quickly, leading to noise
    if dtype is np.float64:
        expected_rate = degree + 1
        tolerance = 0.2
        assert area_rate >= expected_rate - tolerance, (
            f"Area convergence rate {area_rate:.2f} below expected {expected_rate}"
        )
        assert circ_rate >= expected_rate - tolerance, (
            f"Circumference convergence rate {circ_rate:.2f} below expected {expected_rate}"
        )
