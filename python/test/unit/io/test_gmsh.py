# Copyright (C) 2025-2026 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx
import ufl


@pytest.fixture
def gmsh_model():
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cell")
    try:
        yield gmsh.model
    finally:
        gmsh.finalize()


def add_gmsh_cell(model, cell_name, degree):
    """Elevate a discrete reference cell using Gmsh's node ordering."""
    cell_type = getattr(basix.CellType, cell_name.lower())
    vertex_order = {
        "Hexahedron": [0, 1, 3, 2, 4, 5, 7, 6],
        "Prism": [0, 1, 2, 3, 4, 5],
        "Pyramid": [0, 1, 3, 2, 4],
    }[cell_name]
    vertices = basix.geometry(cell_type)[vertex_order]
    entity = model.addDiscreteEntity(3)
    node_tags = np.arange(1, len(vertices) + 1)
    model.mesh.addNodes(3, entity, node_tags, vertices.flatten())
    model.mesh.addElementsByType(entity, model.mesh.getElementType(cell_name, 1), [1], node_tags)
    model.addPhysicalGroup(3, [entity], tag=1)
    model.mesh.setOrder(degree)
    return entity


@pytest.mark.parametrize("cell_name", ["Hexahedron", "Prism", "Pyramid"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_gmsh_cell_ordering(gmsh_model, cell_name, degree):
    """Compare converted nodes with the independent Basix reference element."""
    from dolfinx.io import gmsh as gmshio

    entity = add_gmsh_cell(gmsh_model, cell_name, degree)
    element_types, _, element_nodes = gmsh_model.mesh.getElements(3, entity)
    node_tags, coordinates, _ = gmsh_model.mesh.getNodes()
    nodes = dict(zip(node_tags, coordinates.reshape(-1, 3), strict=True))
    points = np.array([nodes[tag] for tag in element_nodes[0]])

    domain = gmshio.ufl_mesh(element_types[0], 3, np.float64)
    element = domain.ufl_coordinate_element().basix_element
    assert element.degree == degree
    assert element.cell_type.name == cell_name.lower()
    cell_type = getattr(dolfinx.mesh.CellType, cell_name.lower())
    permutation = gmshio.cell_perm_array(cell_type, element.dim)
    np.testing.assert_allclose(points[permutation], element.points, atol=1e-14, rtol=0)

    # Also check the geometry produced by the converted connectivity.
    cells = np.arange(element.dim, dtype=np.int64)[permutation].reshape(1, -1)
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, domain, points)
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
    expected_volume = {"Hexahedron": 1, "Prism": 0.5, "Pyramid": 1 / 3}[cell_name]
    assert np.isclose(volume, expected_volume)


@pytest.mark.parametrize("cell_name", ["Hexahedron", "Prism", "Pyramid"])
def test_third_order_cell_import(gmsh_model, cell_name):
    """Import cubic cells through the complete model-to-mesh path."""
    from dolfinx.io import gmsh as gmshio

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        add_gmsh_cell(gmsh_model, cell_name, 3)

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        return dolfinx.graph.adjacencylist(np.zeros((dual_graph.num_nodes, 1), dtype=np.int32))

    data = gmshio.model_to_mesh(gmsh_model, comm, 0, partitioner=partitioner)
    assert data.mesh.geometry.cmaps[0].degree == 3
    assert data.mesh.topology.index_map(3).size_global == 1
    assert data.cell_tags is not None
    assert np.all(data.cell_tags.values == 1)
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=data.mesh))), op=MPI.SUM
    )
    expected_volume = {"Hexahedron": 1, "Prism": 0.5, "Pyramid": 1 / 3}[cell_name]
    assert np.isclose(volume, expected_volume)


@pytest.mark.parametrize(
    "marker_mode",
    [
        pytest.param(0, marks=pytest.mark.xfail(raises=RuntimeError)),
        pytest.param(1, marks=pytest.mark.xfail(raises=RuntimeError)),
        2,
        pytest.param(3, marks=pytest.mark.xfail(raises=RuntimeError)),
    ],
)
def test_physical_tags(marker_mode) -> None:
    """Test that we catch partially tagged meshes and not tagged
    meshes as errors.
    """
    gmsh = pytest.importorskip("gmsh")

    from dolfinx.io import gmsh as gmshio

    gmsh.initialize()

    def gmsh_tet_model(order):
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model()
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            model.add("Sphere minus box")
            model.setCurrent("Sphere minus box")
            model.occ.addSphere(0, 0, 0, 1)
            model.occ.addSphere(2, 2, 2, 0.3)
            model.occ.synchronize()
            volume_entities = [model[1] for model in model.getEntities(3)]
            volume_entities = volume_entities[:marker_mode]
            for i, entity in enumerate(volume_entities):
                model.addPhysicalGroup(3, [entity], tag=i)
            if marker_mode == 3:  # Check duplicate marker error
                model.addPhysicalGroup(3, [entity], tag=10)
            model.mesh.generate(3)
            gmsh.option.setNumber("General.Terminal", 1)
            model.mesh.setOrder(order)
            gmsh.option.setNumber("General.Terminal", 0)

        mesh_data = gmshio.model_to_mesh(model, comm, 0)
        return mesh_data.mesh, mesh_data.cell_tags

    msh, cell_tags = gmsh_tet_model(1)
    gdim = msh.geometry.dim
    assert msh.geometry.cmaps[0].degree == 1
    assert msh.geometry.dim == gdim
    local_values = np.unique(cell_tags.values)
    all_values = np.unique(np.hstack(msh.comm.allgather(local_values)))
    assert len(all_values) == 2

    gmsh.finalize()
