# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import numpy
import pytest

import dolfin
from dolfin import MPI, Facets
import ufl

from petsc4py import PETSc
from ufl import ds, dx, dS, inner, avg


@pytest.mark.parametrize("mode", [pytest.param(dolfin.cpp.mesh.GhostMode.none,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(dolfin.cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(dolfin.cpp.mesh.GhostMode.shared_vertex,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_ghost_mesh_assembly(mode):
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12,
                                            dolfin.cpp.mesh.CellType.Type.triangle,
                                            mode)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    f = dolfin.Function(V)
    with f.vector().localForm() as f_local:
        f_local.set(10.0)
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds

    # Initial assembly
    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = dolfin.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    assert normA == pytest.approx(0.6713621455570528, rel=1.e-6, abs=1.e-12)

    normb = b.norm()
    assert normb == pytest.approx(1.582294032953906, rel=1.e-6, abs=1.e-12)


@pytest.mark.parametrize("mode", [pytest.param(dolfin.cpp.mesh.GhostMode.none,
                                               marks=pytest.mark.skipif(condition=MPI.size(MPI.comm_world) > 1,
                                                                       reason="Unghosted interior facets fail in parallel")),
                                  pytest.param(dolfin.cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(dolfin.cpp.mesh.GhostMode.shared_vertex,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_ghost_mesh_dS_assembly(mode):
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12,
                                            dolfin.cpp.mesh.CellType.Type.triangle,
                                            mode)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    a = inner(avg(u), avg(v)) * dS

    # Initial assembly
    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    # Check that the norms are the same for all three modes
    normA = A.norm()
    print(normA)

    assert normA == pytest.approx(2.1834054713561906, rel=1.e-6, abs=1.e-12)
