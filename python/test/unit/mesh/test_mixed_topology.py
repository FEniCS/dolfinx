from mpi4py import MPI

import numpy as np

from dolfinx.cpp.log import set_thread_name
from dolfinx.cpp.mesh import Mesh_float64, create_geometry, create_topology
from dolfinx.fem import coordinate_element
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import CellType, GhostMode, create_unit_cube


def test_triquad():
    set_log_level(LogLevel.INFO)

    cells = [[0, 1, 2, 1, 2, 3], [2, 3, 4, 5]]
    orig_index = [[0, 1], [2]]
    ghost_owners = [[], []]
    boundary_vertices = []

    topology = create_topology(
        MPI.COMM_SELF,
        [CellType.triangle, CellType.quadrilateral],
        cells,
        orig_index,
        ghost_owners,
        boundary_vertices,
    )

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
    assert topology.connectivity((2, 0), (0, 0)).num_nodes == 2
    # One quadrlilateral cell
    assert entity_types[2][1] == CellType.quadrilateral
    assert topology.connectivity((2, 1), (0, 0)).num_nodes == 1

    # Create dofmaps for Geometry
    tri = coordinate_element(CellType.triangle, 1)
    quad = coordinate_element(CellType.quadrilateral, 1)
    nodes = [0, 1, 2, 3, 4, 5]
    xdofs = [0, 1, 2, 1, 2, 3, 2, 3, 4, 5]
    x = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0]
    geom = create_geometry(topology, [tri._cpp_object, quad._cpp_object], nodes, xdofs, x, 2)
    print(geom.x)
    print(geom.index_map().size_local)
    print(geom.dofmaps(0))
    print(geom.dofmaps(1))
    set_log_level(LogLevel.WARNING)


def test_mixed_mesh_3d():
    # Mesh = 2 tets, 1 prism, 1 hex, joined.
    cells = [[0, 1, 2, 3, 1, 2, 3, 4], [2, 3, 4, 5, 6, 7], [3, 4, 6, 7, 8, 9, 10, 11]]
    orig_index = [[0, 1], [2], [3]]
    ghost_owners = [[], [], []]
    boundary_vertices = []

    topology = create_topology(
        MPI.COMM_SELF,
        [CellType.tetrahedron, CellType.prism, CellType.hexahedron],
        cells,
        orig_index,
        ghost_owners,
        boundary_vertices,
    )

    entity_types = topology.entity_types
    assert len(entity_types[0]) == 1
    assert len(entity_types[1]) == 1
    assert len(entity_types[2]) == 2
    assert len(entity_types[3]) == 3

    # Create triangle and quadrilateral facets
    topology.create_entities(2)

    qi = topology.entity_types[2].index(CellType.quadrilateral)
    ti = topology.entity_types[2].index(CellType.triangle)

    # Tet -> quad
    assert topology.connectivity((3, 0), (2, qi)) is None
    # Tet -> triangle
    t = topology.connectivity((3, 0), (2, ti))
    assert t.num_nodes == 2
    assert len(t.links(0)) == 4

    # Prism -> quad
    t = topology.connectivity((3, 1), (2, qi))
    assert t.num_nodes == 1
    assert len(t.links(0)) == 3
    # Prism -> triangle
    t = topology.connectivity((3, 1), (2, ti))
    assert t.num_nodes == 1
    assert len(t.links(0)) == 2

    # Hex -> quad
    t = topology.connectivity((3, 2), (2, qi))
    assert t.num_nodes == 1
    assert len(t.links(0)) == 6
    # Hex -> triangle
    assert topology.connectivity((3, 2), (2, ti)) is None

    # Quad -> vertex
    t = topology.connectivity((2, qi), (0, 0))
    assert t.num_nodes == 8
    assert len(t.links(0)) == 4

    # Triangle -> vertex
    t = topology.connectivity((2, ti), (0, 0))
    assert t.num_nodes == 8
    assert len(t.links(0)) == 3

    topology.create_connectivity(2, 1)

    # Quad -> edge
    t = topology.connectivity((2, qi), (1, 0))
    assert t.num_nodes == 8
    assert len(t.links(0)) == 4

    # Tri -> edge
    t = topology.connectivity((2, ti), (1, 0))
    assert t.num_nodes == 8
    assert len(t.links(0)) == 3

    topology.create_connectivity(2, 3)
    # Quad -> prism
    t = topology.connectivity((2, qi), (3, 1))
    assert t.num_nodes == 8
    assert t.array.size == 3

    # Quad -> hex
    t = topology.connectivity((2, qi), (3, 2))
    assert t.num_nodes == 8
    assert t.array.size == 6

    # Tri -> tet
    t = topology.connectivity((2, ti), (3, 0))
    assert t.num_nodes == 8
    assert t.array.size == 8

    # Tri -> prism
    t = topology.connectivity((2, ti), (3, 1))
    assert t.num_nodes == 8
    assert t.array.size == 2


def test_parallel_mixed_mesh():
    rank = MPI.COMM_WORLD.Get_rank()

    # Two triangles and one quadrilateral
    tri = [0, 1, 4, 0, 3, 4]
    quad = [1, 4, 2, 5]
    # cells with global indexing
    cells = [[t + 3 * rank for t in tri], [q + 3 * rank for q in quad]]
    orig_index = [[3 * rank, 1 + 3 * rank], [2 + 3 * rank]]
    # No ghosting
    ghost_owners = [[], []]
    # All vertices are on boundary
    boundary_vertices = [3 * rank + i for i in range(6)]

    topology = create_topology(
        MPI.COMM_WORLD,
        [CellType.triangle, CellType.quadrilateral],
        cells,
        orig_index,
        ghost_owners,
        boundary_vertices,
    )

    # Cell types appear in order as in create_topology
    assert topology.entity_types[2][0] == CellType.triangle
    assert topology.entity_types[2][1] == CellType.quadrilateral

    size = MPI.COMM_WORLD.Get_size()
    assert topology.index_maps(2)[0].size_global == size * 2
    assert topology.index_maps(2)[1].size_global == size

    # Create dofmaps for Geometry
    tri = coordinate_element(CellType.triangle, 1)
    quad = coordinate_element(CellType.quadrilateral, 1)
    nodes = [3 * rank + i for i in range(6)]
    xdofs = np.array([0, 1, 2, 1, 2, 3, 2, 3, 4, 5], dtype=int) + 3 * rank
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
    )
    x[:, 1] += 1.0 * rank
    set_log_level(LogLevel.INFO)
    set_thread_name(str(rank))
    geom = create_geometry(
        topology, [tri._cpp_object, quad._cpp_object], nodes, xdofs, x.flatten(), 2
    )
    print(geom.x)
    print(geom.index_map().size_local)
    print(geom.dofmaps(0))
    print(geom.dofmaps(1))

    mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)
    print(mesh.topology.connectivity((2, 0), (0, 0)))
    print(mesh.topology.connectivity((2, 1), (0, 0)))
    print(mesh.geometry.dim)

    set_log_level(LogLevel.WARNING)


def test_create_entities():
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, CellType.prism, ghost_mode=GhostMode.none)

    # Make triangle and quadrilateral facets
    mesh.topology.create_entities(2)

    assert len(mesh.topology.entity_types[2]) == 2
    qi = mesh.topology.entity_types[2].index(CellType.quadrilateral)
    ti = mesh.topology.entity_types[2].index(CellType.triangle)
    assert qi != ti

    cell_quad = mesh.topology.connectivity((3, 0), (2, qi))
    cell_tri = mesh.topology.connectivity((3, 0), (2, ti))

    assert MPI.COMM_WORLD.allreduce(cell_quad.num_nodes) == 16
    assert len(cell_quad.links(0)) == 3
    assert MPI.COMM_WORLD.allreduce(cell_tri.num_nodes) == 16
    assert len(cell_tri.links(0)) == 2

    quad_v = mesh.topology.connectivity((2, qi), (0, 0))
    tri_v = mesh.topology.connectivity((2, ti), (0, 0))
    ims = mesh.topology.index_maps(2)
    assert len(ims) == 2
    assert ims[qi].size_global == 32
    assert len(quad_v.links(0)) == 4
    assert ims[ti].size_global == 24
    assert len(tri_v.links(0)) == 3

    mesh.topology.create_entities(1)
    # 9 edges on each prism
    cell_edge = mesh.topology.connectivity((3, 0), (1, 0))
    assert cell_edge.links(0).size == 9

    # Triangle and quad to prism (facet->cell)
    mesh.topology.create_connectivity(2, 3)
