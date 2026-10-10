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


@pytest.mark.parametrize(
    "cell_name,vertex_order,expected_volume",
    [
        ("Hexahedron", [0, 1, 3, 2, 4, 5, 7, 6], 1),
        ("Prism", [0, 1, 2, 3, 4, 5], 0.5),
        ("Pyramid", [0, 1, 3, 2, 4], 1 / 3),
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cell_import(gmsh_model, cell_name, vertex_order, expected_volume, degree):
    """Check Gmsh node ordering and import against Basix reference cells."""
    from dolfinx.io import gmsh as gmshio

    model = gmsh_model
    cell_type = getattr(basix.CellType, cell_name.lower())
    vertices = basix.geometry(cell_type)[vertex_order]
    entity = model.addDiscreteEntity(3)
    node_tags = np.arange(1, len(vertices) + 1)
    model.mesh.addNodes(3, entity, node_tags, vertices.flatten())
    model.mesh.addElementsByType(entity, model.mesh.getElementType(cell_name, 1), [1], node_tags)
    model.addPhysicalGroup(3, [entity], tag=1)
    peak = model.addDiscreteEntity(0)
    model.mesh.addElementsByType(peak, 15, [2], [1])
    model.addPhysicalGroup(0, [peak], tag=2)
    model.mesh.setOrder(degree)
    element_types, _, element_nodes = model.mesh.getElements(3, entity)
    node_tags, coordinates, _ = model.mesh.getNodes()
    nodes = dict(zip(node_tags, coordinates.reshape(-1, 3), strict=True))
    points = np.array([nodes[tag] for tag in element_nodes[0]])

    domain = gmshio.ufl_mesh(element_types[0], 3, np.float64)
    element = domain.ufl_coordinate_element().basix_element
    assert element.degree == degree
    assert element.cell_type.name == cell_name.lower()
    cell_type = getattr(dolfinx.mesh.CellType, cell_name.lower())
    permutation = gmshio.cell_perm_array(cell_type, element.dim)
    np.testing.assert_allclose(points[permutation], element.points, atol=1e-14, rtol=0)

    comm = MPI.COMM_WORLD

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        return dolfinx.graph.adjacencylist(np.zeros((dual_graph.num_nodes, 1), dtype=np.int32))

    data = gmshio.model_to_mesh(model, comm, 0, partitioner=partitioner)
    assert data.mesh.geometry.cmaps[0].degree == degree
    assert data.mesh.topology.index_map(3).size_global == 1
    assert data.cell_tags is not None
    assert np.all(data.cell_tags.values == 1)
    assert data.peak_tags is not None
    assert np.all(data.peak_tags.values == 2)
    assert comm.allreduce(len(data.peak_tags.values), op=MPI.SUM) == 1
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=data.mesh))), op=MPI.SUM
    )
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
