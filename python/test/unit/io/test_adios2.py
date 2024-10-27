# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type
from dolfinx.fem import Function, functionspace
from dolfinx.graph import adjacencylist
from dolfinx.mesh import CellType, create_mesh, create_unit_cube, create_unit_square


def generate_mesh(dim: int, simplex: bool, N: int = 5, dtype=None):
    """Helper function for parametrizing over meshes."""
    if dtype is None:
        dtype = default_real_type

    if dim == 2:
        if simplex:
            return create_unit_square(MPI.COMM_WORLD, N, N, dtype=dtype)
        else:
            return create_unit_square(MPI.COMM_WORLD, 2 * N, N, CellType.quadrilateral, dtype=dtype)
    elif dim == 3:
        if simplex:
            return create_unit_cube(MPI.COMM_WORLD, N, N, N, dtype=dtype)
        else:
            return create_unit_cube(MPI.COMM_WORLD, N, N, N, CellType.hexahedron, dtype=dtype)
    else:
        raise RuntimeError("Unsupported dimension")


@pytest.mark.adios2
class TestVTX:
    @pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
    def test_second_order_vtx(self, tempdir):
        from dolfinx.io import VTXWriter

        filename = Path(tempdir, "mesh_vtx.bp")
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=default_real_type)
        cells = np.array([[0, 1, 2]], dtype=np.int32)
        domain = ufl.Mesh(element("Lagrange", "interval", 2, shape=(1,), dtype=default_real_type))
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        with VTXWriter(mesh.comm, filename, mesh) as f:
            f.write(0.0)

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_vtx_mesh(self, tempdir, dim, simplex):
        from dolfinx.io import VTXWriter

        filename = Path(tempdir, "mesh_vtx.bp")
        mesh = generate_mesh(dim, simplex)
        with VTXWriter(mesh.comm, filename, mesh) as f:
            f.write(0.0)
            mesh.geometry.x[:, 1] += 0.1
            f.write(0.1)

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_vtx_functions_fail(self, tempdir, dim, simplex):
        """Test for error when elements differ."""
        from dolfinx.io import VTXWriter

        mesh = generate_mesh(dim, simplex)
        gdim = mesh.geometry.dim
        v = Function(functionspace(mesh, ("Lagrange", 2, (gdim,))))
        w = Function(functionspace(mesh, ("Lagrange", 1)))
        filename = Path(tempdir, "v.bp")
        with pytest.raises(RuntimeError):
            VTXWriter(mesh.comm, filename, [v, w])

    @pytest.mark.parametrize("simplex", [True, False])
    def test_vtx_different_meshes_function(self, tempdir, simplex):
        """Test for error when functions do not share a mesh."""
        from dolfinx.io import VTXWriter

        mesh = generate_mesh(2, simplex)
        v = Function(functionspace(mesh, ("Lagrange", 1)))
        mesh2 = generate_mesh(2, simplex)
        w = Function(functionspace(mesh2, ("Lagrange", 1)))
        filename = Path(tempdir, "v.bp")
        with pytest.raises(RuntimeError):
            VTXWriter(mesh.comm, filename, [v, w])

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_vtx_single_function(self, tempdir, dim, simplex):
        """Test saving a single first order Lagrange functions."""
        from dolfinx.io import VTXWriter

        mesh = generate_mesh(dim, simplex)
        v = Function(functionspace(mesh, ("Lagrange", 1)))

        filename = Path(tempdir, "v.bp")
        writer = VTXWriter(mesh.comm, filename, v)
        writer.write(0)
        writer.close()

        filename = Path(tempdir, "v2.bp")
        writer = VTXWriter(mesh.comm, filename, v._cpp_object)
        writer.write(0)
        writer.close()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_vtx_functions(self, tempdir, dtype, dim, simplex):
        """Test saving high order Lagrange functions."""
        from dolfinx.io import VTXWriter

        xtype = np.real(dtype(0)).dtype
        mesh = generate_mesh(dim, simplex, dtype=xtype)
        gdim = mesh.geometry.dim
        V = functionspace(mesh, ("DG", 2, (gdim,)))
        v = Function(V, dtype=dtype)
        bs = V.dofmap.index_map_bs

        def vel(x):
            values = np.zeros((dim, x.shape[1]), dtype=dtype)
            values[0] = x[1]
            values[1] = x[0]
            return values

        v.interpolate(vel)

        W = functionspace(mesh, ("DG", 2))
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

    def test_save_vtkx_cell_point(self, tempdir):
        """Test writing point-wise data."""
        from dolfinx.io import VTXWriter

        mesh = create_unit_square(MPI.COMM_WORLD, 8, 5)
        P = element("Discontinuous Lagrange", mesh.basix_cell(), 0, dtype=default_real_type)

        V = functionspace(mesh, P)
        u = Function(V)
        u.interpolate(lambda x: 0.5 * x[0])
        u.name = "A"

        filename = Path(tempdir, "v.bp")
        f = VTXWriter(mesh.comm, filename, [u])
        f.write(0)
        f.close()

    def test_empty_rank_mesh(self, tempdir):
        """Test VTXWriter on mesh where some ranks have no cells."""
        from dolfinx.io import VTXWriter

        comm = MPI.COMM_WORLD
        cell_type = CellType.triangle
        domain = ufl.Mesh(
            element("Lagrange", cell_type.name, 1, shape=(2,), dtype=default_real_type)
        )

        def partitioner(comm, nparts, local_graph, num_ghost_nodes):
            """Leave cells on the current rank"""
            dest = np.full(len(cells), comm.rank, dtype=np.int32)
            return adjacencylist(dest)

        if comm.rank == 0:
            cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=default_real_type)
        else:
            cells = np.empty((0, 3), dtype=np.int64)
            x = np.empty((0, 2), dtype=default_real_type)

        mesh = create_mesh(comm, cells, x, domain, partitioner)

        V = functionspace(mesh, ("Lagrange", 1))
        u = Function(V)

        filename = Path(tempdir, "empty_rank_mesh.bp")
        with VTXWriter(comm, filename, u) as f:
            f.write(0.0)

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    @pytest.mark.parametrize("reuse", [True, False])
    def test_vtx_reuse_mesh(self, tempdir, dim, simplex, reuse):
        """Test reusage of mesh by VTXWriter."""
        from dolfinx.io import VTXMeshPolicy, VTXWriter

        adios2 = pytest.importorskip("adios2", minversion="2.10.0")
        if not adios2.is_built_with_mpi:
            pytest.skip("Require adios2 built with MPI support")

        mesh = generate_mesh(dim, simplex)
        v = Function(functionspace(mesh, ("Lagrange", 1)))
        filename = Path(tempdir, "v.bp")
        v.name = "v"
        policy = VTXMeshPolicy.reuse if reuse else VTXMeshPolicy.update

        # Save three steps
        writer = VTXWriter(mesh.comm, filename, v, "BP4", policy)
        writer.write(0)
        v.interpolate(lambda x: 0.5 * x[0])
        writer.write(1)
        v.interpolate(lambda x: x[1])
        writer.write(2)
        writer.close()

        reuse_variables = ["NumberOfEntities", "NumberOfNodes", "connectivity", "geometry", "types"]
        target_all = 3  # For all other variables the step count is number of writes
        target_mesh = 1 if reuse else 3
        # For mesh variables the step count is 1 if reuse else number of writes
        adios = adios2.Adios(comm=mesh.comm)
        io = adios.declare_io("TestData")
        io.set_engine("BP4")
        adios_file = adios2.Stream(io, str(filename), "r", mesh.comm)

        for name, var in adios_file.available_variables().items():
            if name in reuse_variables:
                assert int(var["AvailableStepsCount"]) == target_mesh
            else:
                assert int(var["AvailableStepsCount"]) == target_all
        adios_file.close()
