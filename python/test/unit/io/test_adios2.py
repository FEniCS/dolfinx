# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

import numpy as np
import pytest

import ufl
from dolfinx.common import has_adios2
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import (CellType, create_mesh, create_unit_cube,
                          create_unit_square)

from mpi4py import MPI

try:
    from dolfinx.io import FidesWriter, VTXWriter
except ImportError:
    pytest.skip("Test require ADIOS2", allow_module_level=True)


def generate_mesh(dim: int, simplex: bool, N: int = 5):
    """Helper function for parametrizing over meshes"""
    if dim == 2:
        if simplex:
            return create_unit_square(MPI.COMM_WORLD, N, N)
        else:
            return create_unit_square(MPI.COMM_WORLD, 2 * N, N, CellType.quadrilateral)
    elif dim == 3:
        if simplex:
            return create_unit_cube(MPI.COMM_WORLD, N, N, N)
        else:
            return create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron)
    else:
        raise RuntimeError("Unsupported dimension")


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_fides_mesh(tempdir, dim, simplex):
    """ Test writing of a single Fides mesh with changing geometry"""
    filename = Path(tempdir, "mesh_fides.bp")
    mesh = generate_mesh(dim, simplex)
    with FidesWriter(mesh.comm, filename, mesh) as f:
        f.write(0.0)
        mesh.geometry.x[:, 1] += 0.1
        f.write(0.1)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_two_fides_functions(tempdir, dim, simplex):
    """Test saving two functions with Fides"""
    mesh = generate_mesh(dim, simplex)
    v = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    q = Function(FunctionSpace(mesh, ("Lagrange", 1)))
    filename = Path(tempdir, "v.bp")
    with FidesWriter(mesh.comm, filename, [v._cpp_object, q]) as f:
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
def test_findes_single_function(tempdir, dim, simplex):
    "Test saving a single first order Lagrange functions"
    mesh = generate_mesh(dim, simplex)
    v = Function(FunctionSpace(mesh, ("Lagrange", 1)))
    filename = Path(tempdir, "v.bp")
    writer = FidesWriter(mesh.comm, filename, v)
    writer.write(0)
    writer.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_fides_function_at_nodes(tempdir, dim, simplex):
    """Test saving P1 functions with Fides (with changing geometry)"""
    mesh = generate_mesh(dim, simplex)
    v = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    v.name = "v"
    q = Function(FunctionSpace(mesh, ("Lagrange", 1)))
    q.name = "q"
    filename = Path(tempdir, "v.bp")
    with FidesWriter(mesh.comm, filename, [v, q]) as f:
        for t in [0.1, 0.5, 1]:
            # Only change one function
            q.interpolate(lambda x: t * (x[0] - 0.5)**2)
            f.write(t)

            mesh.geometry.x[:, :2] += 0.1
            if mesh.geometry.dim == 2:
                v.interpolate(lambda x: np.vstack((t * x[0], x[1] + x[1] * 1j)))
            elif mesh.geometry.dim == 3:
                v.interpolate(lambda x: np.vstack((t * x[2], x[0] + x[2] * 2j, x[1])))
            f.write(t)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_second_order_vtx(tempdir):
    filename = Path(tempdir, "mesh_fides.bp")
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    cell = ufl.Cell("interval", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    with VTXWriter(mesh.comm, filename, mesh) as f:
        f.write(0.0)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_mesh(tempdir, dim, simplex):
    filename = Path(tempdir, "mesh_vtx.bp")
    mesh = generate_mesh(dim, simplex)
    with VTXWriter(mesh.comm, filename, mesh) as f:
        f.write(0.0)
        mesh.geometry.x[:, 1] += 0.1
        f.write(0.1)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_functions_fail(tempdir, dim, simplex):
    "Test for error when elements differ"
    mesh = generate_mesh(dim, simplex)
    v = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    w = Function(FunctionSpace(mesh, ("Lagrange", 1)))
    filename = Path(tempdir, "v.bp")
    with pytest.raises(RuntimeError):
        VTXWriter(mesh.comm, filename, [v, w])


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_different_meshes_function(tempdir, simplex):
    "Test for error when functions do not share a mesh"
    mesh = generate_mesh(2, simplex)
    v = Function(FunctionSpace(mesh, ("Lagrange", 1)))
    mesh2 = generate_mesh(2, simplex)
    w = Function(FunctionSpace(mesh2, ("Lagrange", 1)))
    filename = Path(tempdir, "v.bp")
    with pytest.raises(RuntimeError):
        VTXWriter(mesh.comm, filename, [v, w])


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_single_function(tempdir, dim, simplex):
    "Test saving a single first order Lagrange functions"
    mesh = generate_mesh(dim, simplex)
    v = Function(FunctionSpace(mesh, ("Lagrange", 1)))

    filename = Path(tempdir, "v.bp")
    writer = VTXWriter(mesh.comm, filename, v)
    writer.write(0)
    writer.close()

    filename = Path(tempdir, "v2.bp")
    writer = VTXWriter(mesh.comm, filename, v._cpp_object)
    writer.write(0)
    writer.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_vtx_functions(tempdir, dtype, dim, simplex):
    "Test saving high order Lagrange functions"
    mesh = generate_mesh(dim, simplex)
    V = VectorFunctionSpace(mesh, ("DG", 2))
    v = Function(V, dtype=dtype)
    bs = V.dofmap.index_map_bs

    def vel(x):
        values = np.zeros((dim, x.shape[1]), dtype=dtype)
        values[0] = x[1]
        values[1] = x[0]
        return values
    v.interpolate(vel)

    W = FunctionSpace(mesh, ("DG", 2))
    w = Function(W, dtype=v.dtype)
    w.interpolate(lambda x: x[0] + x[1])

    filename = Path(tempdir, f"v-{np.dtype(dtype).num}.bp")
    f = VTXWriter(mesh.comm, filename, [v, w])

    # Set two cells to 0
    for c in [0, 1]:
        dofs = np.asarray([V.dofmap.cell_dofs(c) * bs + b for b in range(bs)], dtype=np.int32)
        v.x.array[dofs] = 0
        w.x.array[W.dofmap.cell_dofs(c)] = 1
    v.x.scatter_forward()
    w.x.scatter_forward()

    # Save twice and update geometry
    for t in [0.1, 1]:
        mesh.geometry.x[:, :2] += 0.1
        f.write(t)

    f.close()


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_save_vtkx_cell_point(tempdir):
    """Test writing point-wise data"""
    mesh = create_unit_square(MPI.COMM_WORLD, 8, 5)
    P = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)

    V = FunctionSpace(mesh, P)
    u = Function(V)
    u.interpolate(lambda x: 0.5 * x[0])
    u.name = "A"

    filename = Path(tempdir, "v.bp")
    with pytest.raises(RuntimeError):
        f = VTXWriter(mesh.comm, filename, [u])
        f.write(0)
        f.close()


def test_empty_rank_mesh(tempdir):
    """Test VTXWriter on mesh where some ranks have no cells"""
    comm = MPI.COMM_WORLD
    cell_type = CellType.triangle
    domain = ufl.Mesh(
        ufl.VectorElement("Lagrange", ufl.Cell(cell_type.name), 1))

    def partitioner(comm, nparts, local_graph, num_ghost_nodes):
        """Leave cells on the current rank"""
        dest = np.full(len(cells), comm.rank, dtype=np.int32)
        return create_adjacencylist(dest)

    if comm.rank == 0:
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        cells = create_adjacencylist(cells)
        x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    else:
        cells = create_adjacencylist(np.empty((0, 3), dtype=np.int64))
        x = np.empty((0, 2), dtype=np.float64)

    mesh = create_mesh(comm, cells, x, domain, partitioner)

    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)

    filename = Path(tempdir, "empty_rank_mesh.bp")
    with VTXWriter(comm, filename, u) as f:
        f.write(0.0)
