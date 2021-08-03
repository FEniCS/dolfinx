# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfinx import Function, FunctionSpace, io  # VectorFunctionSpace,
from dolfinx.cpp.io import FidesWriter, has_adios2
from dolfinx.cpp.mesh import CellType
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_save_mesh(tempdir):
    filename = os.path.join(tempdir, "mesh.bp")

    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    f = FidesWriter(mesh.mpi_comm(), filename, io.mode.write, mesh)
    f.write(0.0)
    f.close()


def generate_mesh(dim: int, simplex: bool, N: int = 3):
    if dim == 2:
        if simplex:
            return UnitSquareMesh(MPI.COMM_WORLD, N, N)
        else:
            return UnitSquareMesh(MPI.COMM_WORLD, N, N, CellType.quadrilateral)
    elif dim == 3:
        if simplex:
            return UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        else:
            return UnitCubeMesh(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
    else:
        raise RuntimeError("Unsupported dimension")


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_save_function(tempdir, dim, simplex):
    mesh = generate_mesh(dim, simplex)
    # V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    # v = Function(V)

    Q = FunctionSpace(mesh, ("Lagrange", 1))
    q = Function(Q)

    filename = os.path.join(tempdir, "v.bp")
    # f = FidesWriter(mesh.mpi_comm(), filename, io.mode.write, [v._cpp_object, q._cpp_object])
    f = FidesWriter(mesh.mpi_comm(), filename, io.mode.write, [q._cpp_object])

    for t in [0.1, 1]:
        q.interpolate(lambda x: t * (x[0] - 0.5)**2)
        mesh.geometry.x[:, :2] += 0.1
        # if mesh.geometry.dim == 2:
        #     v.interpolate(lambda x: (t * x[0], x[1] + x[1] * 1j))
        # elif mesh.geometry.dim == 3:
        #     v.interpolate(lambda x: (t * x[2], x[0] + x[2] * 2j, x[1]))
        f.write(t)

    f.close()
