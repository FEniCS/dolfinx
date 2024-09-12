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
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.graph import adjacencylist
from dolfinx.mesh import CellType, GhostMode, create_mesh, create_unit_cube, create_unit_square


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


# TODO: Fix problems with ("HDF5", ".h5"), ("BP4", ".bp"),
@pytest.mark.adios2
@pytest.mark.parametrize("encoder, suffix", [("BP5", ".bp")])
@pytest.mark.parametrize("ghost_mode", [GhostMode.shared_facet, GhostMode.none])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_mesh_read_write(encoder, suffix, ghost_mode, dtype, dim, simplex, tmp_path):
    "Test writing of a mesh"
    from dolfinx.io import ADIOS2, read_mesh, write_mesh

    N = 5
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_mesh_{encoder}"

    mesh = generate_mesh(dim, simplex, N, dtype)

    adios = ADIOS2(mesh.comm)
    tag = "mesh-write"
    adios.add_io(filename=str(file.with_suffix(suffix)), tag=tag, engine_type=encoder, mode="write")

    write_mesh(adios, tag, mesh)

    adios_read = ADIOS2(MPI.COMM_WORLD)
    tag = "mesh-read"
    adios_read.add_io(
        filename=str(file.with_suffix(suffix)), tag=tag, engine_type=encoder, mode="read"
    )

    mesh_adios = read_mesh(adios_read, tag, MPI.COMM_WORLD, ghost_mode=ghost_mode)

    mesh_adios.comm.Barrier()
    mesh.comm.Barrier()

    for i in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(i)
        mesh_adios.topology.create_entities(i)
        assert (
            mesh.topology.index_map(i).size_global == mesh_adios.topology.index_map(i).size_global
        )

    # Check that integration over different entities are consistent
    measures = [ufl.ds, ufl.dx] if ghost_mode is GhostMode.none else [ufl.ds, ufl.dS, ufl.dx]
    for measure in measures:
        c_adios = assemble_scalar(form(1 * measure(domain=mesh_adios), dtype=dtype))
        c_ref = assemble_scalar(form(1 * measure(domain=mesh), dtype=dtype))
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )


# TODO: Fix problems with ("HDF5", ".h5"), ("BP4", ".bp"),
@pytest.mark.adios2
@pytest.mark.parametrize("encoder, suffix", [("BP5", ".bp")])
@pytest.mark.parametrize("ghost_mode", [GhostMode.shared_facet, GhostMode.none])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_timedep_mesh_read_write(encoder, suffix, ghost_mode, dtype, dim, simplex, tmp_path):
    "Test writing of a time dependent mesh"
    from dolfinx.io import ADIOS2, read_mesh, read_timestamps, update_mesh, write_mesh

    N = 5
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_timedep_mesh_{encoder}"

    mesh = generate_mesh(dim, simplex, N, dtype)

    def displace(x):
        return np.asarray(
            [
                x[0] + 0.1 * np.sin(x[0]) * np.cos(x[1]),
                x[1] + 0.6 * np.cos(x[0]) * np.sin(x[1]),
                x[2],
            ]
        )

    adios = ADIOS2(mesh.comm)
    tag_write = "mesh-write"
    adios.add_io(
        filename=str(file.with_suffix(suffix)), tag=tag_write, engine_type=encoder, mode="write"
    )

    # Write mesh
    write_mesh(adios, tag_write, mesh, time=0.0)

    delta_x1 = displace(mesh.geometry.x.T).T
    mesh.geometry.x[:] += delta_x1

    write_mesh(adios, tag_write, mesh, time=1.0)

    delta_x2 = displace(mesh.geometry.x.T).T
    mesh.geometry.x[:] += delta_x2

    write_mesh(adios, tag_write, mesh, time=2.0)

    adios.close(tag_write)

    # reset mesh geometry to the one at time=0.0
    mesh.geometry.x[:] -= delta_x1 + delta_x2

    tag_rra = "mesh-readrandomaccess"
    adios.add_io(
        filename=str(file.with_suffix(suffix)),
        tag=tag_rra,
        engine_type=encoder,
        mode="readrandomaccess",
    )
    times = read_timestamps(adios, tag_rra)
    adios.close(tag_rra)
    assert np.all(np.isclose(times, [0.0, 1.0, 2.0]))

    tag_read = "mesh-read"
    adios.add_io(
        filename=str(file.with_suffix(suffix)), tag=tag_read, engine_type=encoder, mode="read"
    )
    mesh_adios = read_mesh(adios, tag_read, MPI.COMM_WORLD, ghost_mode=ghost_mode)

    mesh_adios.comm.Barrier()
    mesh.comm.Barrier()

    # Check that integration over different entities are consistent
    measures = [ufl.ds, ufl.dx] if ghost_mode is GhostMode.none else [ufl.ds, ufl.dS, ufl.dx]
    for step, time in enumerate(times):
        if step == 1:
            mesh.geometry.x[:] += delta_x1
        if step == 2:
            mesh.geometry.x[:] += delta_x2

        # FIXME: update_mesh at time time=0.0 should work!?
        if step > 0:
            update_mesh(adios, tag_read, mesh_adios, step)

        mesh_adios.comm.Barrier()
        mesh.comm.Barrier()

        for measure in measures:
            c_adios = assemble_scalar(form(1 * measure(domain=mesh_adios), dtype=dtype))
            c_ref = assemble_scalar(form(1 * measure(domain=mesh), dtype=dtype))
            assert np.isclose(
                mesh_adios.comm.allreduce(c_adios, MPI.SUM),
                mesh.comm.allreduce(c_ref, MPI.SUM),
            )


@pytest.mark.adios2
class TestFides:
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_fides_mesh(self, tempdir, dim, simplex):
        """Test writing of a single Fides mesh with changing geometry."""
        from dolfinx.io import FidesWriter

        filename = Path(tempdir, "mesh_fides.bp")
        mesh = generate_mesh(dim, simplex)
        with FidesWriter(mesh.comm, filename, mesh) as f:
            f.write(0.0)
            mesh.geometry.x[:, 1] += 0.1
            f.write(0.1)

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_two_fides_functions(self, tempdir, dim, simplex):
        """Test saving two functions with Fides."""
        from dolfinx.io import FidesWriter

        mesh = generate_mesh(dim, simplex)
        gdim = mesh.geometry.dim
        v = Function(functionspace(mesh, ("Lagrange", 1, (gdim,))))
        q = Function(functionspace(mesh, ("Lagrange", 1)))
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

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_fides_single_function(self, tempdir, dim, simplex):
        """Test saving a single first order Lagrange functions."""
        from dolfinx.io import FidesWriter

        mesh = generate_mesh(dim, simplex)
        v = Function(functionspace(mesh, ("Lagrange", 1)))
        filename = Path(tempdir, "v.bp")
        writer = FidesWriter(mesh.comm, filename, v)
        writer.write(0)
        writer.close()

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("simplex", [True, False])
    def test_fides_function_at_nodes(self, tempdir, dim, simplex):
        """Test saving P1 functions with Fides (with changing geometry)."""
        from dolfinx.io import FidesWriter

        mesh = generate_mesh(dim, simplex)
        gdim = mesh.geometry.dim
        v = Function(functionspace(mesh, ("Lagrange", 1, (gdim,))), dtype=default_scalar_type)
        v.name = "v"
        q = Function(functionspace(mesh, ("Lagrange", 1)))
        q.name = "q"
        filename = Path(tempdir, "v.bp")
        if np.issubdtype(default_scalar_type, np.complexfloating):
            alpha = 1j
        else:
            alpha = 0

        with FidesWriter(mesh.comm, filename, [v, q]) as f:
            for t in [0.1, 0.5, 1]:
                # Only change one function
                q.interpolate(lambda x: t * (x[0] - 0.5) ** 2)
                f.write(t)

                mesh.geometry.x[:, :2] += 0.1
                if mesh.geometry.dim == 2:
                    v.interpolate(lambda x: np.vstack((t * x[0], x[1] + x[1] * alpha)))
                elif mesh.geometry.dim == 3:
                    v.interpolate(lambda x: np.vstack((t * x[2], x[0] + x[2] * 2 * alpha, x[1])))
                f.write(t)


@pytest.mark.adios2
class TestVTX:
    @pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
    def test_second_order_vtx(self, tempdir):
        from dolfinx.io import VTXWriter

        filename = Path(tempdir, "mesh_fides.bp")
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
