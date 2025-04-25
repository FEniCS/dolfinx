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
from dolfinx.io.vtkhdf import read_mesh, write_mesh
from dolfinx.mesh import CellType, Mesh, create_unit_cube, create_unit_square


def test_read_write_vtkhdf_mesh2d():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float32)
    write_mesh("example2d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float32)
    assert mesh2.geometry.x.dtype == np.float32
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float64)
    assert mesh2.geometry.x.dtype == np.float64
    assert mesh.topology.index_map(2).size_global == mesh2.topology.index_map(2).size_global


def test_read_write_vtkhdf_mesh3d():
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type=CellType.prism)
    write_mesh("example3d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example3d.vtkhdf")

    assert mesh.topology.index_map(3).size_global == mesh2.topology.index_map(3).size_global


def test_read_write_mixed_topology(mixed_topology_mesh):
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

    part = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    mesh = dolfinx.cpp.mesh.create_mesh(
        MPI.COMM_WORLD, cells_np, [e._cpp_object for e in coordinate_elements], geom, part
    )
    py_mesh = Mesh(mesh, None)

    # Write mesh to file
    write_mesh("mixed_mesh_second_order.vtkhdf", py_mesh)

    # Read mesh as a 2D grid and a flat manifold in 3D
    for gdim in [2, 3]:
        mesh_in = read_mesh(MPI.COMM_WORLD, "mixed_mesh_second_order.vtkhdf", gdim=gdim)
        assert mesh_in.geometry.dim == gdim
        assert mesh_in.geometry.index_map().size_global == 12
        cmap_0 = mesh_in.geometry._cpp_object.cmaps(0)
        cmap_1 = mesh_in.geometry._cpp_object.cmaps(1)
        assert cmap_0.degree == 2
        assert cmap_1.degree == 2

        cell_types = mesh.topology.cell_types
        assert dolfinx.mesh.CellType.quadrilateral in cell_types
        assert dolfinx.mesh.CellType.triangle in cell_types


@pytest.mark.parametrize("order", [1, 2, 3])
def test_read_write_higher_order_mesh(order):
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
    # Read in mesh with gmsh to create reference dat
    ref_mesh = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, rank).mesh
    gmsh.finalize()

    ref_volume_form = dolfinx.fem.form(
        1 * ufl.dx(domain=ref_mesh), form_compiler_options={"scalar_type": ref_mesh.geometry.x.dtype}
    )
    ref_volume = comm.allreduce(dolfinx.fem.assemble_scalar(ref_volume_form), op=MPI.SUM)

    ref_surface_form = dolfinx.fem.form(
        1 * ufl.ds(domain=ref_mesh), form_compiler_options={"scalar_type": ref_mesh.geometry.x.dtype}
    )
    ref_surface = comm.allreduce(dolfinx.fem.assemble_scalar(ref_surface_form), op=MPI.SUM)

    # Write to file
    filename = f"gmsh_{order}_order_sphere.vtkhdf"
    write_mesh(filename, ref_mesh)
    del ref_mesh, ref_volume_form

    # Read mesh
    mesh = read_mesh(comm, filename)

    # Compare surface and volume metrics
    volume_form = dolfinx.fem.form(
        1 * ufl.dx(domain=mesh), form_compiler_options={"scalar_type": ref_mesh.geometry.x.dtype}
    )
    volume = comm.allreduce(dolfinx.fem.assemble_scalar(volume_form), op=MPI.SUM)
    assert np.isclose(ref_volume, volume)

    surface_form = dolfinx.fem.form(
        1 * ufl.ds(domain=mesh), form_compiler_options={"scalar_type": ref_mesh.geometry.x.dtype}
    )
    surface = comm.allreduce(dolfinx.fem.assemble_scalar(surface_form), op=MPI.SUM)
    assert np.isclose(ref_surface, surface)
