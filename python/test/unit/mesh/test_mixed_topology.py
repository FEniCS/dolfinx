from mpi4py import MPI
from dolfinx.cpp.mesh import create_topology, CellType
import numpy as np
import pytest


@pytest.mark.skip_in_parallel
def test_triquad():
    cells = [[0, 1, 2, 1, 2, 3], [2, 3, 4, 5]]
    orig_index = [[0, 1], [2]]
    ghost_owners = [[], []]
    boundary_vertices = []

    topology = create_topology(MPI.COMM_WORLD, [CellType.triangle, CellType.quadrilateral],
                               cells, orig_index, ghost_owners, boundary_vertices)

    maps = topology.index_maps(topology.dim)
    assert len(maps) == 2
    # Two triangles and one quad
    assert maps[0].size_local == 2
    assert maps[1].size_local == 1

    # Six vertices in map
    map0 = topology.index_maps(0)
    assert len(map0) == 1
    assert map0[0].size_local == 6

    entity_types = topology.entity_types
    assert len(entity_types[0]) == 1
    assert len(entity_types[1]) == 1
    assert len(entity_types[2]) == 2
    assert CellType.interval in entity_types[1]
    # Two triangle cells
    assert entity_types[2][0] == CellType.triangle
    assert topology.connectivity(2, 0, 0, 0).num_nodes == 2
    # One quadrlilateral cell
    assert entity_types[2][1] == CellType.quadrilateral
    assert topology.connectivity(2, 1, 0, 0).num_nodes == 1


@pytest.mark.skip_in_parallel
def test_mixed_mesh_3d():
    cells = [[0, 1, 2, 3, 1, 2, 3, 4], [2, 3, 4, 5, 6, 7], [3, 4, 6, 7, 8, 9, 10, 11]]
    orig_index = [[0, 1], [2], [3]]
    ghost_owners = [[], [], []]
    boundary_vertices = []

    topology = create_topology(MPI.COMM_WORLD, [CellType.tetrahedron, CellType.prism, CellType.hexahedron],
                               cells, orig_index, ghost_owners, boundary_vertices)

    entity_types = topology.entity_types
    assert len(entity_types[0]) == 1
    assert len(entity_types[1]) == 1
    assert len(entity_types[2]) == 2
    assert len(entity_types[3]) == 3
    assert CellType.triangle in entity_types[2]
    assert CellType.quadrilateral in entity_types[2]


@pytest.mark.skip_in_parallel
def test_prism():
    cells = [[0, 1, 2, 3, 4, 5]]
    orig_index = [[0]]
    ghost_owners = [[]]
    boundary_vertices = []

    topology = create_topology(MPI.COMM_WORLD, [CellType.prism], cells, orig_index, ghost_owners, boundary_vertices)

    print(topology.connectivity(3, 0))
    print(topology.entity_types)


@pytest.mark.skip_in_serial
def test_parallel_mixed_mesh():
    rank = MPI.COMM_WORLD.Get_rank()

    tri = [0, 1, 4, 0, 3, 4]
    quad = [1, 4, 2, 5]
    # cells with global indexing
    cells = [[t + 3 * rank for t in tri], [q + 3 * rank for q in quad]]
    orig_index = [[3 * rank, 1 + 3 * rank], [2 + 3 * rank]]
    ghost_owners = [[], []]
    boundary_vertices = [3 * rank + i for i in range(6)]

    topology = create_topology(MPI.COMM_WORLD, [CellType.triangle, CellType.quadrilateral],
                               cells, orig_index, ghost_owners, boundary_vertices)

    assert topology.entity_types[2][0] == CellType.triangle
    assert topology.entity_types[2][1] == CellType.quadrilateral
    print(topology.connectivity(2, 0, 0, 0))
    print(topology.connectivity(2, 1, 0, 0))

    w = topology.index_map(0)
    print(w.local_to_global(np.arange(6, dtype=int)))

    size = MPI.COMM_WORLD.Get_size()
    assert topology.index_maps(2)[0].size_global == size * 2
    assert topology.index_maps(2)[1].size_global == size

    print(topology.index_maps(2)[0].size_global)
