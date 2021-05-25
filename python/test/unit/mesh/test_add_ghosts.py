import os

import dolfinx
import dolfinx.io
import numpy as np
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI
import ufl
assert(tempdir)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1)


def facets_to_cells(mesh, facets):
    tdim = mesh.topology.dim
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    if len(facets) == 0:
        return np.asarray([], dtype=np.int32)
    else:
        return np.hstack([facet_to_cell.links(facet) for facet in facets])


def create_layer_partition(N, comm, tempdir):
    """
    Create a NxN UnitSquareMesh partitioned in stripes.
    Inspired by test_custom_partitioners
    """

    # Create original mesh and write to file
    mesh = dolfinx.UnitSquareMesh(comm, N, N, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral,
                                  ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

    filename = os.path.join(tempdir, "tmp_mesh_.xdmf")
    with dolfinx.io.XDMFFile(comm, filename, "w") as file:
        file.write_mesh(mesh)

    # Read all geometry data on all processes
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, filename, "r") as file:
        x_global = file.read_geometry_data()

    # Read topology data
    with dolfinx.io.XDMFFile(comm, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        x = file.read_geometry_data()
        topo = file.read_topology_data()

    num_local_coor = x.shape[0]
    all_sizes = comm.allgather(num_local_coor)
    all_sizes.insert(0, 0)
    all_ranges = np.cumsum(all_sizes)

    # Partition mesh in layers, capture geometrical data and topological
    # data from outer scope
    # Testing the premise: coordinates are read contiguously in chunks
    rank = comm.rank
    assert (np.all(x_global[all_ranges[rank]:all_ranges[rank + 1]] == x))
    cell = ufl.Cell(dolfinx.cpp.mesh.to_string(cell_shape))
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_degree))

    def partitioner(*args):
        midpoints = np.mean(x_global[topo], axis=1)
        dest = np.floor((midpoints[:, 0] * comm.size) % comm.size).astype(np.int32)
        return dolfinx.cpp.graph.AdjacencyList_int32(dest)

    ghost_mode = dolfinx.cpp.mesh.GhostMode.none
    new_mesh = dolfinx.mesh.create_mesh(comm, topo, x, domain, ghost_mode, partitioner)
    new_mesh.topology.create_connectivity_all()
    return new_mesh


def test_add_ghosts_left_to_right(tempdir):
    """
    Add ghost cells for all cells with a facet on the left side of domain to all processes with cells that
    have facets on the right side of the domain.
    """
    comm = MPI.COMM_WORLD
    # Make sure no cell on either side is ghost on the other and that it can be ran with any number of processes
    N = max(comm.size, 8)

    # Extract some mesh variables
    mesh = create_layer_partition(N, comm, tempdir)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = cell_map.size_local

    # Locate facets on left and right side of mesh
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right)

    # Map the facets on each side to their corresponding cell (keep local only)
    num_cells_local = mesh.topology.index_map(tdim).size_local
    left_cells = facets_to_cells(mesh, left_facets)
    left_cells_local = left_cells[left_cells < num_cells_local]

    right_cells = facets_to_cells(mesh, right_facets)
    right_cells_local = right_cells[right_cells < num_cells_local]

    # Send which processes has local cells with facets on right side to all processes
    has_right = np.full(comm.size, int(len(right_cells_local) > 0))
    procs_with_right = np.flatnonzero(np.array(comm.alltoall(has_right), dtype=np.int32))

    # Get local cells that are ghosted on other processes (before repartitioning)
    # NOTE: In C++ we can use shared_indices and avoid communication
    shared_indices = cell_map.compute_shared_indices()

    # Create adjacency process list for each local cell.
    data = []
    offset = [0]
    for i in range(num_cells_local):
        # Add owning process
        local_ranks = [comm.rank]
        # Add processes from ghosts already in the mesh
        old_ghosts = shared_indices.get(i)
        if old_ghosts is not None:
            for ghost in old_ghosts:
                if ghost not in local_ranks:
                    local_ranks.append(ghost)
        # For each local cell that connects to the left boundary, add all
        # processes that has a cell connecting to the right boundary
        if i in left_cells_local:
            for rank in procs_with_right:
                if not (rank in local_ranks) and rank != comm.rank:
                    local_ranks.append(rank)

        data.extend(local_ranks)
        offset.append(len(data))
    cell_partitioning = dolfinx.cpp.graph.AdjacencyList_int32(data, offset)
    # for i in range(cell_partitioning.num_nodes):
    #     print(comm.rank, i, cell_partitioning.links(i))

    # Create new mesh with additional ghosts
    new_mesh = dolfinx.mesh.add_ghosts(mesh, cell_partitioning)
    num_new_ghosts_local = N - len(left_cells)
    num_ghosts_old = mesh.topology.index_map(tdim).num_ghosts
    num_ghosts = new_mesh.topology.index_map(tdim).num_ghosts
    if len(right_cells_local) > 0:
        print(num_ghosts, num_ghosts_old, num_new_ghosts_local)
        assert(num_ghosts - num_ghosts_old == num_new_ghosts_local)
    # Output for number ghosts
    V = dolfinx.FunctionSpace(new_mesh, ("DG", 0))
    u = dolfinx.Function(V)
    # u.x.array[:] = 1 + comm.rank
    u.vector.array[:] = 1 + comm.rank
    dolfinx.cpp.la.scatter_reverse(u.x, dolfinx.cpp.common.ScatterMode.add)

    with dolfinx.io.XDMFFile(comm, "new_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(new_mesh)
        xdmf.write_function(u)
