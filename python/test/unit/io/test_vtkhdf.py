# Copyright (C) 2024-2025 Chris Richardson and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import ufl
from dolfinx.io.vtkhdf import read_mesh, write_cell_data, write_mesh, write_point_data
from dolfinx.mesh import CellType, Mesh, create_unit_cube, create_unit_square


def test_read_write_vtkhdf_mesh2d() -> None:
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float32)
    write_mesh("example2d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float32)
    assert mesh2.geometry.x.dtype == np.float32
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float64)
    assert mesh2.geometry.x.dtype == np.float64
    assert mesh.topology.index_map(2).size_global == mesh2.topology.index_map(2).size_global


def test_read_write_vtkhdf_mesh3d() -> None:
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type=CellType.prism)
    write_mesh("example3d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example3d.vtkhdf")

    assert mesh.topology.index_map(3).size_global == mesh2.topology.index_map(3).size_global


@pytest.mark.parametrize("num_threads", [1, 4])
def test_read_write_vtkhdf_num_threads(num_threads) -> None:
    filename = "example_num_threads.vtkhdf"
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 5)
    write_mesh(filename, mesh)

    mesh_1 = read_mesh(MPI.COMM_WORLD, filename, num_threads=1)
    mesh_n = read_mesh(MPI.COMM_WORLD, filename, num_threads=num_threads)

    for m in (mesh_1, mesh_n):
        assert (
            m.topology.index_map(m.topology.dim).size_global
            == mesh.topology.index_map(mesh.topology.dim).size_global
        )
        assert m.topology.index_map(0).size_global == mesh.topology.index_map(0).size_global

    vol_1 = mesh_1.comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(1 * ufl.dx(domain=mesh_1), dtype=mesh_1.geometry.x.dtype)
        ),
        op=MPI.SUM,
    )
    vol_n = mesh_n.comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(1 * ufl.dx(domain=mesh_n), dtype=mesh_n.geometry.x.dtype)
        ),
        op=MPI.SUM,
    )
    assert np.isclose(vol_1, vol_n)


def test_read_vtkhdf_num_threads_invalid() -> None:
    filename = "example_num_threads_invalid.vtkhdf"
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    write_mesh(filename, mesh)

    with pytest.raises(ValueError):
        read_mesh(MPI.COMM_WORLD, filename, num_threads=0)


def test_read_write_mixed_topology(mixed_topology_mesh) -> None:
    mesh = Mesh(mixed_topology_mesh, None)
    write_mesh("mixed_mesh.vtkhdf", mesh)

    mesh2 = read_mesh(MPI.COMM_WORLD, "mixed_mesh.vtkhdf", np.float64)
    for t in mesh2.topology.entity_types[-1]:
        assert t in mesh.topology.entity_types[-1]


def test_read_write_higher_order():
    # Create a simple, 2 cell mesh consisting of a second order quadrilateral and
    # a second order triangle.
    geom = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0],
            [1, 0.5],
            [0.5, 1],
            [0, 0.5],
            [0.5, 0.5],
            [2.0, 0],
            [1.5, -0.2],
            [1.5, 0.6],
        ],
        dtype=np.float64,
    )
    # Nodes ordered as VTK
    if MPI.COMM_WORLD.rank == 0:
        topology_quad = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
        topology_tri = np.array([[1, 9, 2, 10, 11, 5]], dtype=np.int64)

    else:
        topology_quad = np.empty((0, 9), dtype=np.int64)
        topology_tri = np.empty((0, 6), dtype=np.int64)

    quad_perm = dolfinx.io.utils.cell_perm_vtk(dolfinx.mesh.CellType.quadrilateral, 9)
    tri_perm = dolfinx.io.utils.cell_perm_vtk(dolfinx.mesh.CellType.triangle, 6)
    topology_quad = topology_quad[:, quad_perm]
    topology_tri = topology_tri[:, tri_perm]

    cells_np = [topology_quad.flatten(), topology_tri.flatten()]
    coordinate_elements = [
        dolfinx.fem.coordinate_element(cell, 2)
        for cell in [dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.CellType.triangle]
    ]

    max_cells_per_facet = 2
    part = dolfinx.graph.partitioner()
    mesh = dolfinx.cpp.mesh._create_mixed_mesh(
        MPI.COMM_WORLD,
        cells_np,
        [e._cpp_object for e in coordinate_elements],
        geom,
        part,
        dolfinx.mesh.GhostMode.none,
        max_cells_per_facet,
        num_threads=1,
        cell_weights=None,
        reorder_fn=None,
    )
    py_mesh = Mesh(mesh, None)

    # Write mesh to file
    write_mesh("mixed_mesh_second_order.vtkhdf", py_mesh)

    # Read mesh as a 2D grid and a flat manifold in 3D
    for gdim in [2, 3]:
        mesh_in = read_mesh(MPI.COMM_WORLD, "mixed_mesh_second_order.vtkhdf", gdim=gdim)
        assert mesh_in.geometry.dim == gdim
        assert mesh_in.geometry.index_map().size_global == 12
        cmap_0 = mesh_in.geometry.cmaps[0]
        cmap_1 = mesh_in.geometry.cmaps[1]
        assert cmap_0.degree == 2
        assert cmap_1.degree == 2

        cell_types = mesh.topology.cell_types
        assert dolfinx.mesh.CellType.quadrilateral in cell_types
        assert dolfinx.mesh.CellType.triangle in cell_types


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_higher_order_mesh(order, dtype) -> None:
    try:
        import gmsh
    except ImportError:
        pytest.skip()

    # Create a tetrahedral mesh of a sphere
    res = 0.3
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    rank = 0
    model = None
    gmsh.model.add(f"mesh_{order}")
    if comm.rank == rank:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.occ.addSphere(0, 0, 0, 1, tag=1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)
    comm.Barrier()

    model = comm.bcast(model, root=rank)
    # Use the same precision for the reference and the mesh read back.
    ref_mesh = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, rank, dtype=dtype).mesh
    gmsh.finalize()

    # File cell indices follow rank order, excluding ghost cells.
    num_cells = ref_mesh.topology.index_map(3).size_local
    cell_geometry = ref_mesh.geometry.x[ref_mesh.geometry.dofmaps[0][:num_cells]]
    ref_cell_geometry = np.concatenate(comm.allgather(cell_geometry))

    ref_volume_form = dolfinx.fem.form(
        1 * ufl.dx(domain=ref_mesh),
        dtype=ref_mesh.geometry.x.dtype,
    )
    ref_volume = comm.allreduce(dolfinx.fem.assemble_scalar(ref_volume_form), op=MPI.SUM)

    ref_surface_form = dolfinx.fem.form(
        1 * ufl.ds(domain=ref_mesh),
        dtype=ref_mesh.geometry.x.dtype,
    )
    ref_surface = comm.allreduce(dolfinx.fem.assemble_scalar(ref_surface_form), op=MPI.SUM)

    # Write to file
    filename = f"gmsh_{order}_order_sphere_{np.dtype(dtype).name}.vtkhdf"
    write_mesh(filename, ref_mesh)
    del ref_mesh, ref_volume_form

    # Check both mesh construction thread counts.
    mesh = read_mesh(comm, filename, dtype=dtype, num_threads=1)
    mesh_mt = read_mesh(comm, filename, dtype=dtype, num_threads=4)

    for m in (mesh, mesh_mt):
        domain = m.ufl_domain()
        assert domain is not None
        assert m.geometry.x.dtype == dtype
        assert domain.ufl_coordinate_element().basix_element.dtype == dtype
        assert m.geometry.cmaps[0].degree == order
        np.testing.assert_array_equal(
            m.geometry.x[m.geometry.dofmaps[0]],
            ref_cell_geometry[m.topology.original_cell_index],
        )

    assert (
        mesh.topology.index_map(mesh.topology.dim).size_global
        == mesh_mt.topology.index_map(mesh_mt.topology.dim).size_global
    )
    assert mesh.topology.index_map(0).size_global == mesh_mt.topology.index_map(0).size_global

    volume_mt_form = dolfinx.fem.form(1 * ufl.dx(domain=mesh_mt), dtype=mesh_mt.geometry.x.dtype)
    volume_mt = comm.allreduce(dolfinx.fem.assemble_scalar(volume_mt_form), op=MPI.SUM)
    surface_mt_form = dolfinx.fem.form(1 * ufl.ds(domain=mesh_mt), dtype=mesh_mt.geometry.x.dtype)
    surface_mt = comm.allreduce(dolfinx.fem.assemble_scalar(surface_mt_form), op=MPI.SUM)

    # Assembly can accumulate in a different order after repartitioning.
    rtol = 100 * np.finfo(dtype).eps

    volume_form = dolfinx.fem.form(1 * ufl.dx(domain=mesh), dtype=mesh.geometry.x.dtype)
    volume = comm.allreduce(dolfinx.fem.assemble_scalar(volume_form), op=MPI.SUM)
    assert np.isclose(ref_volume, volume, rtol=rtol, atol=0)

    surface_form = dolfinx.fem.form(1 * ufl.ds(domain=mesh), dtype=mesh.geometry.x.dtype)
    surface = comm.allreduce(dolfinx.fem.assemble_scalar(surface_form), op=MPI.SUM)
    assert np.isclose(ref_surface, surface, rtol=rtol, atol=0)

    assert np.isclose(volume, volume_mt, rtol=rtol, atol=0)
    assert np.isclose(surface, surface_mt, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_point_data(dtype) -> None:
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype)
    filename = "point_data.vtkhdf"
    write_mesh(filename, mesh)
    point_data = np.arange(mesh.geometry.index_map().size_local)
    for j in range(3):
        write_point_data(filename, mesh, point_data, float(j))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("width", [1, 3])
def test_write_cell_data(dtype, width) -> None:
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype)
    filename = "cell_data.vtkhdf"
    write_mesh(filename, mesh)
    cell_data = np.arange(mesh.topology.index_map(2).size_local * width)
    for j in range(3):
        write_cell_data(filename, mesh, cell_data, float(j))


def test_write_mixed_topology_data(mixed_topology_mesh) -> None:
    mesh = Mesh(mixed_topology_mesh, None)
    filename = "mixed_point_data.vtkhdf"
    write_mesh(filename, mesh)
    point_data = np.arange(mesh.geometry.index_map().size_local, dtype=np.float64)
    for j in range(10):
        write_point_data(filename, mesh, point_data, float(j))
        point_data *= 0.9

    filename = "mixed_cell_data.vtkhdf"
    write_mesh(filename, mesh)
    b = sum([im.size_local for im in mesh.topology.index_maps(mesh.topology.dim)])
    cell_data = np.arange(b, dtype=np.float64)
    write_cell_data(filename, mesh, cell_data, 0.0)


@pytest.mark.parametrize("degree", [1, 2])
def test_read_write_prism(degree, tempdir) -> None:
    """Full quadratic prisms retain their coordinate element and geometry."""
    from pathlib import Path

    import basix
    import basix.ufl

    element = basix.ufl.element(
        "Lagrange", "prism", degree, basix.LagrangeVariant.equispaced, shape=(3,), dtype=np.float64
    )
    comm = MPI.COMM_WORLD
    # A connected stack avoids partitioner corner cases for a single cell.
    points = np.concatenate(
        [element.basix_element.points + np.array([0, 0, i]) for i in range(2 * comm.size)]
    )
    points[:, 0] += 0.1 * points[:, 2] ** 2
    points, indices = np.unique(points, axis=0, return_inverse=True)
    cells = indices.reshape(-1, element.basix_element.dim).astype(np.int64)
    if comm.rank != 0:
        cells = np.empty((0, element.basix_element.dim), dtype=np.int64)
        points = np.empty((0, 3), dtype=np.float64)
    mesh = dolfinx.mesh.create_mesh(comm, cells, ufl.Mesh(element), points)
    num_cells = mesh.topology.index_map(3).size_local
    geometry = np.concatenate(comm.allgather(mesh.geometry.x[mesh.geometry.dofmaps[0][:num_cells]]))
    filename = Path(tempdir, "prism.vtkhdf")
    write_mesh(filename, mesh)
    restored = read_mesh(comm, filename)
    assert restored.geometry.cmaps[0].degree == degree
    np.testing.assert_array_equal(
        restored.geometry.x[restored.geometry.dofmaps[0]],
        geometry[restored.topology.original_cell_index],
    )


def test_write_quadratic_pyramid_rejected(tempdir) -> None:
    """Unsupported pyramids must not be labelled as linear VTK cells."""
    from pathlib import Path

    import basix
    import basix.ufl

    element = basix.ufl.element(
        "Lagrange", "pyramid", 2, basix.LagrangeVariant.equispaced, shape=(3,), dtype=np.float64
    )
    comm = MPI.COMM_WORLD
    points = element.basix_element.points
    cells = np.arange(element.basix_element.dim, dtype=np.int64).reshape(1, -1)
    if comm.rank != 0:
        cells = np.empty((0, element.basix_element.dim), dtype=np.int64)
        points = np.empty((0, 3), dtype=np.float64)

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        return dolfinx.graph.adjacencylist(np.zeros((dual_graph.num_nodes, 1), dtype=np.int32))

    mesh = dolfinx.mesh.create_mesh(comm, cells, ufl.Mesh(element), points, partitioner=partitioner)
    with pytest.raises(ValueError, match="Lagrange pyramids are not implemented in VTK"):
        write_mesh(Path(tempdir, "pyramid.vtkhdf"), mesh)
    with pytest.raises(ValueError, match="Lagrange pyramids are not implemented in VTK"):
        dolfinx.plot.vtk_mesh(mesh)
    with dolfinx.io.VTKFile(comm, Path(tempdir, "pyramid.pvd"), "w") as vtk:
        with pytest.raises(ValueError, match="Lagrange pyramids are not implemented in VTK"):
            vtk.write_mesh(mesh)
