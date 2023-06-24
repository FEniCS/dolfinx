# Copyright (C) 2018-2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import pytest

import ufl
from dolfinx import fem
from dolfinx.fem import Function, FunctionSpace, form
from dolfinx.mesh import GhostMode, create_unit_square
from ufl import avg, inner

from mpi4py import MPI
from petsc4py import PETSc


def dx_from_ufl(mesh):
    return ufl.dx


def ds_from_ufl(mesh):
    return ufl.ds


def dS_from_ufl(mesh):
    return ufl.dS


@pytest.mark.parametrize("mode",
                         [GhostMode.none, GhostMode.shared_facet,
                          pytest.param(GhostMode.shared_vertex,
                                       marks=pytest.mark.xfail(reason="Shared vertex currently disabled"))])
@pytest.mark.parametrize("dx", [dx_from_ufl])
@pytest.mark.parametrize("ds", [ds_from_ufl])
def test_ghost_mesh_assembly(mode, dx, ds):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = dx(mesh)
    ds = ds(mesh)

    f = Function(V)
    f.x.array[:] = 10.0
    a = form(inner(f * u, v) * dx + inner(u, v) * ds)
    L = form(inner(f, v) * dx + inner(2.0, v) * ds)

    # Initial assembly
    A = fem.petsc.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    assert normA == pytest.approx(0.6713621455570528, rel=1.e-5, abs=1.e-8)

    normb = b.norm()
    assert normb == pytest.approx(1.582294032953906, rel=1.e-5, abs=1.e-8)

    A.destroy()
    b.destroy()


@pytest.mark.parametrize("mode",
                         [GhostMode.shared_facet,
                          pytest.param(GhostMode.none,
                                       marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size > 1,
                                                                reason="Unghosted interior facets fail in parallel")),
                             pytest.param(GhostMode.shared_vertex,
                                          marks=pytest.mark.xfail(
                                              reason="Shared vertex currently disabled"))])
@pytest.mark.parametrize("dS", [dS_from_ufl])
def test_ghost_mesh_dS_assembly(mode, dS):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dS = dS(mesh)
    a = form(inner(avg(u), avg(v)) * dS)

    # Initial assembly
    A = fem.petsc.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    assert normA == pytest.approx(2.1834054713561906, rel=1.e-5, abs=1.e-12)
    A.destroy()
