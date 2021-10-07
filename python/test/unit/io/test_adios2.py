# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
import ufl
from dolfinx import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.cpp.io import FidesWriter, VTXWriter, has_adios2
from dolfinx.cpp.mesh import CellType
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)


@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test should only be run in serial.")
@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_second_order_fides(tempdir):
    """Check that fides throws error on second order mesh"""
    filename = os.path.join(tempdir, "mesh_fides.bp")
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0.5, 0, 0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    cell = ufl.Cell("interval", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    with pytest.raises(RuntimeError):
        FidesWriter(mesh.mpi_comm(), filename, mesh)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_functions_from_different_meshes_fides(tempdir):
    """Check that the underlying ADIOS2Writer catches sending in
    functions on different meshes"""
    filename = os.path.join(tempdir, "mesh_fides.bp")
    mesh0 = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    mesh1 = UnitSquareMesh(MPI.COMM_WORLD, 10, 2)
    V0, V1 = FunctionSpace(mesh0, ("Lagrange", 1)), FunctionSpace(mesh1, ("Lagrange", 1))
    u0, u1 = Function(V0), Function(V1)
    with pytest.raises(RuntimeError):
        FidesWriter(mesh0.mpi_comm(), filename, [u0._cpp_object, u1._cpp_object])


def generate_mesh(dim: int, simplex: bool, N: int = 3):
    """Helper function for parametrizing over meshes"""
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
def test_fides_mesh(tempdir, dim, simplex):
    """ Test writing of a single Fides mesh with changing geometry"""
    filename = os.path.join(tempdir, "mesh_fides.bp")
    mesh = generate_mesh(dim, simplex)
    f = FidesWriter(mesh.mpi_comm(), filename, mesh)
    f.write(0.0)
    mesh.geometry.x[:, 1] += 0.1
    f.write(0.1)
    f.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_mixed_fides_functions(tempdir, dim, simplex):
    """Test saving CG-2 and CG-1 functions with Fides"""
    mesh = generate_mesh(dim, simplex)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    v = Function(V)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    q = Function(Q)

    filename = os.path.join(tempdir, "v.bp")
    with pytest.raises(RuntimeError):
        f = FidesWriter(mesh.mpi_comm(), filename, [v._cpp_object, q._cpp_object])


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_two_fides_functions(tempdir, dim, simplex):
    """Test saving two functions with Fides"""
    mesh = generate_mesh(dim, simplex)
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    v = Function(V)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    q = Function(Q)

    filename = os.path.join(tempdir, "v.bp")
    f = FidesWriter(mesh.mpi_comm(), filename, [v._cpp_object, q._cpp_object])
    f.write(0)

    def vel(x):
        values = np.zeros((dim, x.shape[1]))
        values[0] = x[1]
        values[1] = x[0]
        return values
    v.interpolate(vel)
    q.interpolate(lambda x: x[0])
    f.write(1)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_fides_function_at_nodes(tempdir, dim, simplex):
    """Test saving CG-1 functions with Fides (with changing geometry)"""
    mesh = generate_mesh(dim, simplex)
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    v = Function(V)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    q = Function(Q)

    filename = os.path.join(tempdir, "v.bp")
    f = FidesWriter(mesh.mpi_comm(), filename, [v._cpp_object, q._cpp_object])
    for t in [0.1, 0.5, 1]:
        # Only change one function
        q.interpolate(lambda x: t * (x[0] - 0.5)**2)
        f.write(t)

        mesh.geometry.x[:, :2] += 0.1
        if mesh.geometry.dim == 2:
            v.interpolate(lambda x: (t * x[0], x[1] + x[1] * 1j))
        elif mesh.geometry.dim == 3:
            v.interpolate(lambda x: (t * x[2], x[0] + x[2] * 2j, x[1]))
        f.write(t)
    f.close()


@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test should only be run in serial.")
@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_second_order_vtx(tempdir):
    filename = os.path.join(tempdir, "mesh_fides.bp")

    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0.5, 0, 0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    cell = ufl.Cell("interval", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    f = VTXWriter(mesh.mpi_comm(), filename, mesh)
    f.write(0.0)
    f.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_mesh(tempdir, dim, simplex):
    filename = os.path.join(tempdir, "mesh_vtx.bp")
    mesh = generate_mesh(dim, simplex)
    f = VTXWriter(mesh.mpi_comm(), filename, mesh)
    f.write(0.0)
    mesh.geometry.x[:, 1] += 0.1
    f.write(0.1)
    f.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_functions(tempdir, dim, simplex):
    "Test saving high order Lagrange functions"
    mesh = generate_mesh(dim, simplex)
    V = FunctionSpace(mesh, ("DG", 2))
    v = Function(V)
    filename = os.path.join(tempdir, "v.bp")
    f = VTXWriter(mesh.mpi_comm(), filename, [v._cpp_object])
    v.interpolate(lambda x: x[0] + x[1])
    for c in [0, 1]:
        v.x.array[V.dofmap.cell_dofs(c)] = 0
    for t in [0.1, 1]:
        mesh.geometry.x[:, :2] += 0.1
        f.write(t)
    f.close()
