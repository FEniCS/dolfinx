from mpi4py import MPI

import numpy as np

import basix
from dolfinx.fem import coordinate_element
from dolfinx.mesh import CellType, create_mesh


def test_prism_mesh():
    cells = [np.arange(6)]
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    elem = coordinate_element(CellType.prism, 1)
    mesh = create_mesh(MPI.COMM_WORLD, cells, elem, x.flatten())
    assert mesh.topology.index_map(0).size_local == 6
    mesh.topology.create_entities(1)
    assert mesh.topology.index_map(1).size_local == 9
    mesh.topology.create_entities(2)
    assert mesh.topology.index_maps(2)[0].size_local == 3
    assert mesh.topology.index_maps(2)[1].size_local == 2


def test_quadratic_prism_mesh():
    # Work out geometry for quadratic cell
    layout = basix.cell.sub_entity_connectivity(basix.cell.CellType.prism)
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    x_edge = np.array([sum(x[edge[0]]) / 2.0 for edge in layout[1]])
    x_facet = np.array([sum(x[facet[0]]) / len(facet[0]) for facet in layout[2]])
    x = np.concatenate((x, x_edge, x_facet))
    cells = [np.arange(18, dtype=int)]
    elem = coordinate_element(CellType.prism, 2)
    mesh = create_mesh(MPI.COMM_WORLD, cells, elem, x.flatten())
    assert mesh.topology.index_map(0).size_local == 6
    assert mesh.topology.index_map(1).size_local == 9
    assert mesh.topology.index_maps(2)[0].size_local == 3
    assert mesh.topology.index_maps(2)[1].size_local == 2
