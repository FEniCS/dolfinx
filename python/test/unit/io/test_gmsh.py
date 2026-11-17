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
def gmsh_session():
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


@pytest.fixture
def gmsh_model(gmsh_session):
    gmsh_session.model.add("cell")
    return gmsh_session.model


# Gmsh element family, the Gmsh vertex order on the Basix reference
# cell, and the highest degree Gmsh defines
_gmsh_cells = {
    "interval": ("Line", [0, 1], 10),
    "triangle": ("Triangle", [0, 1, 2], 10),
    "quadrilateral": ("Quadrangle", [0, 1, 3, 2], 10),
    "tetrahedron": ("Tetrahedron", [0, 1, 2, 3], 10),
    "hexahedron": ("Hexahedron", [0, 1, 3, 2, 4, 5, 7, 6], 9),
    "prism": ("Prism", [0, 1, 2, 3, 4, 5], 9),
    "pyramid": ("Pyramid", [0, 1, 3, 2, 4], 9),
}


@pytest.mark.parametrize(
    "cell_name,degree",
    [
        (cell, k)
        for cell, (_, _, max_degree) in _gmsh_cells.items()
        for k in range(1, max_degree + 1)
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cell_import(gmsh_model, cell_name, degree, dtype):
    """Check the Gmsh node ordering and the import of every Gmsh cell.

    One Gmsh element is placed on the Basix reference vertices and raised
    to the degree with ``setOrder``, which puts its nodes on the Basix
    equispaced points. Permuted to DOLFINx order, and in the mesh read
    with ``model_to_mesh``, they must be the points of the equispaced
    Basix element.
    """
    from dolfinx.io import gmsh as gmshio

    model = gmsh_model
    family, vertex_order, _ = _gmsh_cells[cell_name]
    basix_cell = getattr(basix.CellType, cell_name)
    vertices = basix.geometry(basix_cell)[vertex_order]
    tdim = vertices.shape[1]
    x = np.zeros((len(vertices), 3))
    x[:, :tdim] = vertices
    entity = model.addDiscreteEntity(tdim)
    node_tags = np.arange(1, len(vertices) + 1)
    model.mesh.addNodes(tdim, entity, node_tags, x.flatten())
    model.mesh.addElementsByType(entity, model.mesh.getElementType(family, 1), [1], node_tags)
    model.addPhysicalGroup(tdim, [entity], tag=1)
    if tdim == 3:
        peak = model.addDiscreteEntity(0)
        model.mesh.addElementsByType(peak, model.mesh.getElementType("Point", 1), [2], [1])
        model.addPhysicalGroup(0, [peak], tag=2)
    model.mesh.setOrder(degree)
    element_types, _, element_nodes = model.mesh.getElements(tdim, entity)
    assert element_types[0] == model.mesh.getElementType(family, degree)
    assert gmshio._gmsh_to_cells[element_types[0]] == (cell_name, degree)
    all_node_tags, coordinates, _ = model.mesh.getNodes()
    nodes = dict(zip(all_node_tags, coordinates.reshape(-1, 3), strict=True))
    points = np.array([nodes[tag][:tdim] for tag in element_nodes[0]])

    element = basix.create_element(
        basix.ElementFamily.P, basix_cell, degree, basix.LagrangeVariant.equispaced
    )
    cell_type = getattr(dolfinx.mesh.CellType, cell_name)
    permutation = gmshio.cell_perm_array(cell_type, element.dim)
    np.testing.assert_allclose(points[permutation], element.points, atol=1e-12, rtol=0)

    comm = MPI.COMM_WORLD

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        return dolfinx.graph.adjacencylist(np.zeros((dual_graph.num_nodes, 1), dtype=np.int32))

    data = gmshio.model_to_mesh(model, comm, 0, partitioner=partitioner, dtype=dtype)
    mesh = data.mesh
    assert mesh.geometry.x.dtype == dtype
    assert mesh.geometry.cmaps[0].degree == degree
    eps = np.finfo(dtype).eps
    assert mesh.topology.index_map(tdim).size_global == 1
    assert data.cell_tags is not None
    assert np.all(data.cell_tags.values == 1)
    if tdim == 3:
        assert data.peak_tags is not None
        assert np.all(data.peak_tags.values == 2)
        assert comm.allreduce(len(data.peak_tags.values), op=MPI.SUM) == 1

    # The nodes of the cell in the mesh, in DOLFINx order
    if mesh.topology.index_map(tdim).size_local == 1:
        cell_nodes = mesh.geometry.x[mesh.geometry.dofmaps[0][0], :tdim]
        np.testing.assert_allclose(cell_nodes, element.points, atol=10 * eps, rtol=0)

    # JIT-compiling a form for high-degree geometry is slow
    if degree <= 3:
        volume = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh), dtype=dtype)),
            op=MPI.SUM,
        )
        assert np.isclose(volume, basix.cell.volume(basix_cell), rtol=100 * eps)


@pytest.mark.parametrize(
    "marker_mode",
    [
        pytest.param(0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1, marks=pytest.mark.xfail(raises=ValueError)),
        2,
        pytest.param(3, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_physical_tags(gmsh_session, marker_mode) -> None:
    """Test that we catch partially tagged meshes and not tagged
    meshes as errors.
    """
    gmsh = gmsh_session

    from dolfinx.io import gmsh as gmshio

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
