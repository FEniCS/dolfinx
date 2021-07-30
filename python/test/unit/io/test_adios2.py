# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfinx import Function, VectorFunctionSpace, FunctionSpace
from dolfinx.cpp.io import ADIOS2File, has_adios2

from dolfinx.cpp.mesh import CellType
from dolfinx.common import has_petsc_complex
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_save_mesh(tempdir):
    filename = os.path.join(tempdir, "mesh.bp")

    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    f = ADIOS2File(mesh.mpi_comm(), filename, "w")
    f.write_mesh(mesh)
    f.close()


def generate_mesh(dim: int, simplex: bool):
    if dim == 2:
        if simplex:
            return UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
        else:
            return UnitSquareMesh(MPI.COMM_WORLD, 5, 5, CellType.quadrilateral)
    elif dim == 3:
        if simplex:
            return UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5)
        else:
            return UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5, CellType.hexahedron)
    else:
        raise RuntimeError("Unsupported dimension")


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_save_function(tempdir, dim, simplex):
    mesh = generate_mesh(dim, simplex)
    V = VectorFunctionSpace(mesh, ("CG", 1))
    v = Function(V)
    if mesh.geometry.dim == 2:
        v.interpolate(lambda x: (x[0], x[1] + x[1] * 1j * has_petsc_complex))
    elif mesh.geometry.dim == 3:
        v.interpolate(lambda x: (x[2], x[0] + x[2] * 2j * has_petsc_complex, x[1]))
    Q = FunctionSpace(mesh, ("CG", 1))
    q = Function(Q)
    q.interpolate(lambda x: (x[0] - 0.5)**2)
    filename = os.path.join(tempdir, "v.bp")
    f = ADIOS2File(mesh.mpi_comm(), filename, "w")
    f.write_mesh(mesh)
    f.write_function([v._cpp_object, q._cpp_object])
    f.close()
