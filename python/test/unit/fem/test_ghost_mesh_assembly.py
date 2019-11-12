# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import pytest
from petsc4py import PETSc

import ufl
from dolfin import MeshFunction, MPI, fem
from dolfin.cpp.mesh import GhostMode
from dolfin.function import Function
from dolfin.functionspace import FunctionSpace
from dolfin.generation import UnitSquareMesh
from ufl import avg, inner, Measure


def dx_from_ufl(mesh):
    return ufl.dx


def dx_from_measure(mesh):
    subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, 1)
    dx = Measure("dx")(subdomain_data=subdomains, domain=mesh)
    dx = dx(1)
    return dx


def dx_from_measure_and_subdomain(mesh):
    dx = dx_from_measure(mesh)
    return dx(1)


def ds_from_ufl(mesh):
    return ufl.ds


def ds_from_measure(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 1)
    ds = Measure("ds")(subdomain_data=boundaries, domain=mesh)
    return ds


def ds_from_measure_and_subdomain(mesh):
    ds = ds_from_measure(mesh)
    return ds(1)


def dS_from_ufl(mesh):
    return ufl.dS


def dS_from_measure(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 1)
    dS = Measure("dS")(subdomain_data=boundaries, domain=mesh)
    return dS


def dS_from_measure_and_subdomain(mesh):
    dS = dS_from_measure(mesh)
    return dS(1)


@pytest.mark.parametrize("mode",
                         [pytest.param(GhostMode.none),
                          pytest.param(GhostMode.shared_facet,
                                       marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                               reason="Shared ghost modes fail in serial")),
                          pytest.param(GhostMode.shared_vertex,
                                       marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                               reason="Shared ghost modes fail in serial"))])
@pytest.mark.parametrize("dx",
                         [dx_from_ufl, dx_from_measure, dx_from_measure_and_subdomain])
@pytest.mark.parametrize("ds",
                         [ds_from_ufl, ds_from_measure, ds_from_measure_and_subdomain])
def test_ghost_mesh_assembly(mode, dx, ds):
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = dx(mesh)
    ds = ds(mesh)

    f = Function(V)
    with f.vector.localForm() as f_local:
        f_local.set(10.0)
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds

    # Initial assembly
    A = fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    assert normA == pytest.approx(0.6713621455570528, rel=1.e-6, abs=1.e-12)

    normb = b.norm()
    assert normb == pytest.approx(1.582294032953906, rel=1.e-6, abs=1.e-12)


@pytest.mark.parametrize("mode",
                         [pytest.param(GhostMode.none,
                                       marks=pytest.mark.skipif(condition=MPI.size(MPI.comm_world) > 1,
                                                                reason="Unghosted interior facets fail in parallel")),
                          pytest.param(GhostMode.shared_facet,
                                       marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                               reason="Shared ghost modes fail in serial")),
                          pytest.param(GhostMode.shared_vertex,
                                       marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                               reason="Shared ghost modes fail in serial"))])
@pytest.mark.parametrize("dS",
                         [dS_from_ufl, dS_from_measure, dS_from_measure_and_subdomain])
def test_ghost_mesh_dS_assembly(mode, dS):
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dS = dS(mesh)

    a = inner(avg(u), avg(v)) * dS

    # Initial assembly
    A = fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    print(normA)

    assert normA == pytest.approx(2.1834054713561906, rel=1.e-6, abs=1.e-12)
