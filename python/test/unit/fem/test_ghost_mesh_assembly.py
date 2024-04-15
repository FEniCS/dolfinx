# Copyright (C) 2018-2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import fem, la
from dolfinx.fem import Function, form, functionspace
from dolfinx.mesh import GhostMode, create_unit_square
from ufl import avg, inner


def dx_from_ufl(mesh):
    return ufl.dx


def ds_from_ufl(mesh):
    return ufl.ds


def dS_from_ufl(mesh):
    return ufl.dS


@pytest.mark.parametrize(
    "mode",
    [
        GhostMode.none,
        GhostMode.shared_facet,
        pytest.param(
            GhostMode.shared_vertex,
            marks=pytest.mark.xfail(reason="Shared vertex currently disabled"),
        ),
    ],
)
@pytest.mark.parametrize("dx", [dx_from_ufl])
@pytest.mark.parametrize("ds", [ds_from_ufl])
def test_ghost_mesh_assembly(mode, dx, ds):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx, ds = dx(mesh), ds(mesh)

    f = Function(V)
    f.x.array[:] = 10.0
    a = form(inner(f * u, v) * dx + inner(u, v) * ds)
    L = form(inner(f, v) * dx + inner(2.0, v) * ds)

    # Initial assembly
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    b = fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    assert isinstance(b, la.Vector)

    # Check that the norms are the same for all three modes
    normA = np.sqrt(A.squared_norm())
    assert normA == pytest.approx(0.6713621455570528, rel=1.0e-5, abs=1.0e-8)
    normb = la.norm(b)
    assert normb == pytest.approx(1.582294032953906, rel=1.0e-5, abs=1.0e-8)


@pytest.mark.parametrize(
    "mode",
    [
        GhostMode.shared_facet,
        pytest.param(
            GhostMode.none,
            marks=pytest.mark.skipif(
                condition=MPI.COMM_WORLD.size > 1,
                reason="Unghosted interior facets fail in parallel",
            ),
        ),
        pytest.param(
            GhostMode.shared_vertex,
            marks=pytest.mark.xfail(reason="Shared vertex currently disabled"),
        ),
    ],
)
@pytest.mark.parametrize("dS", [dS_from_ufl])
def test_ghost_mesh_dS_assembly(mode, dS):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dS = dS(mesh)
    a = form(inner(avg(u), avg(v)) * dS)

    # Initial assembly
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)

    # Check that the norms are the same for all three modes
    normA = np.sqrt(A.squared_norm())
    assert normA == pytest.approx(2.1834054713561906, rel=1.0e-5, abs=1.0e-12)
