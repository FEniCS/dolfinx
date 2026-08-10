# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Mesh partitioning
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_partition.py>`
# * {download}`Jupyter notebook <./demo_partition.ipynb>`
# ```
#
# This demo illustrates how to:
# - Create a mesh from cell and geometry data that is already distributed
#   across MPI ranks
# - Select the partitioner that distributes the cells
# - Measure the quality of the resulting partition
#
# ## Partitioners
#
# When {py:func}`create_mesh <dolfinx.mesh.create_mesh>` is called, the
# input cells are redistributed across ranks by a *partitioner*. The
# partitioner decides which rank will own each cell, and the choice
# involves a trade-off between the cost of computing the partition and its
# quality.
#
# Two families are used here:
#
# - **Graph partitioners** work on the mesh *dual graph*, in which each
#   cell is a node and cells sharing a facet are connected by an edge.
#   They aim to divide the nodes into equally sized parts while cutting as
#   few edges as possible. They are used via
#   {py:func}`create_cell_partitioner
#   <dolfinx.mesh.create_cell_partitioner>`. Three are available,
#   depending on how DOLFINx was built: ParMETIS `Kway` (multilevel k-way
#   partitioning), PT-SCOTCH and KaHIP.
# - **Geometric partitioners** work on the *positions* of the cells. Cells
#   are ordered along a space-filling curve through the mesh and the curve
#   is cut into equal pieces. This is much cheaper than graph
#   partitioning, and the cost barely grows with the number of ranks, but
#   more edges are cut. These are used via
#   {py:func}`create_geometric_cell_partitioner
#   <dolfinx.mesh.create_geometric_cell_partitioner>`, which computes a
#   coordinate for each cell (the mean of its vertex positions) and passes
#   it to the partitioner.
#
#   ParMETIS provides two: `GeomKway`, which uses the space-filling curve
#   to redistribute the graph and then applies k-way partitioning to it,
#   and `Geom`, which uses the curve alone. DOLFINx also provides its own
#   curve partitioner that requires no external library.
#
# Which of ParMETIS, PT-SCOTCH and KaHIP are present depends on the build,
# so the demo checks {py:data}`dolfinx.has_parmetis`,
# {py:data}`dolfinx.has_ptscotch` and {py:data}`dolfinx.has_kahip` and
# uses those that are available.
#
# ## Measures of partition quality
#
# Two quantities are reported for each partition:
#
# - **Imbalance**, the largest number of cells owned by any rank divided
#   by the average. A perfectly balanced partition has an imbalance of 1.
#   Work per rank in an assembly loop is proportional to its cell count,
#   so the slowest rank sets the pace.
# - **Edge cut**, the number of facets that are shared by two cells owned
#   by different ranks. Each cut facet is a place where data must be
#   communicated, so a smaller cut means less communication later. Cut
#   facets are exactly the mesh 'inter-process' facets, so the cut can be
#   read off the created mesh. Note that this requires the mesh to be
#   built with ghost cells, i.e. with `GhostMode.shared_facet`.

# +
from mpi4py import MPI

import numpy as np

from dolfinx import graph, has_kahip, has_parmetis, has_ptscotch
from dolfinx.fem import coordinate_element
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_cell_partitioner,
    create_geometric_cell_partitioner,
    create_mesh,
)

# -

# ## Distributed input data
#
# Mesh data is often read from file in parallel, with each rank holding an
# arbitrary block of the cells and of the geometry. Here the same
# situation is created without a file: each rank builds the block of a
# structured cube mesh of tetrahedra that corresponds to its slice of the
# global cell range. Every cell appears on exactly one rank, and the
# vertex indices are global, so the blocks together describe one mesh.


def cube_block(comm: MPI.Comm, n: int):
    """Local block of the cells and points of a unit cube of tetrahedra.

    Each hexahedral cell of an ``n x n x n`` grid is split into six
    tetrahedra. Cells and points are shared out over the ranks of
    ``comm`` in contiguous blocks.

    Args:
        comm: MPI communicator to distribute the mesh data over.
        n: Number of divisions in each direction.

    Returns:
        Cell vertices (global indices) with shape ``(num_cells, 4)``, and
        point coordinates with shape ``(num_points, 3)``.
    """
    # Cells: this rank's slice of the n^3 hexahedra
    c0 = (n**3 * comm.rank) // comm.size
    c1 = (n**3 * (comm.rank + 1)) // comm.size
    i = np.arange(c0, c1, dtype=np.int64)
    iz, j = np.divmod(i, n * n)
    iy, ix = np.divmod(j, n)

    # The eight corners of each hexahedron, in lexicographic vertex
    # numbering
    v = [
        (iz + dz) * (n + 1) ** 2 + (iy + dy) * (n + 1) + ix + dx
        for dz in (0, 1)
        for dy in (0, 1)
        for dx in (0, 1)
    ]
    tets = [(0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4), (0, 3, 2, 7), (0, 6, 4, 7), (0, 2, 6, 7)]
    cells = np.stack([v[k] for t in tets for k in t], axis=1).reshape(-1, 4)

    # Points: this rank's slice of the (n + 1)^3 grid points
    p0 = ((n + 1) ** 3 * comm.rank) // comm.size
    p1 = ((n + 1) ** 3 * (comm.rank + 1)) // comm.size
    p = np.arange(p0, p1, dtype=np.int64)
    pz, q = np.divmod(p, (n + 1) ** 2)
    py, px = np.divmod(q, n + 1)
    x = np.stack([px, py, pz], axis=1).astype(np.float64) / n

    return np.ascontiguousarray(cells), np.ascontiguousarray(x)


# ## Partition quality
#
# The imbalance is computed from the number of cells owned by each rank.
# The edge cut is the global number of inter-process facets: a facet is
# listed as inter-process on the rank that owns it when its two cells are
# owned by different ranks, so summing the local counts counts each cut
# facet once.


def partition_quality(msh) -> tuple[float, int]:
    """Compute the imbalance and edge cut of a mesh partition.

    Args:
        msh: Mesh to measure.

    Returns:
        Cell imbalance (largest over average number of owned cells) and
        the global number of cut facets.
    """
    comm = msh.comm
    tdim = msh.topology.dim

    cell_map = msh.topology.index_map(tdim)
    max_owned = comm.allreduce(cell_map.size_local, MPI.MAX)
    imbalance = max_owned * comm.size / cell_map.size_global

    # Inter-process facets are available once facets and the facet-to-cell
    # connectivity have been computed
    msh.topology.create_entities(tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    cut = comm.allreduce(len(msh.topology.interprocess_facets()), MPI.SUM)

    return imbalance, cut


# ## Creating a mesh with each partitioner
#
# The cell and geometry blocks and the coordinate element are the same in
# each case; only the partitioner differs. Note that the geometric
# partitioners are given the same coordinate array `x` that is passed to
# {py:func}`create_mesh <dolfinx.mesh.create_mesh>`, as they need the cell
# positions to partition on.

# +
comm = MPI.COMM_WORLD
n = 24
cells, x = cube_block(comm, n)
cmap = coordinate_element(CellType.tetrahedron, 1)
ghost_mode = GhostMode.shared_facet

partitioners = {}

if has_parmetis:
    partitioners["ParMETIS Kway"] = create_cell_partitioner(
        graph.partitioner_parmetis(), ghost_mode, 2
    )
    for label, method in [
        ("ParMETIS GeomKway", graph.ParMETISGeomMethod.kway),
        ("ParMETIS Geom", graph.ParMETISGeomMethod.curve),
    ]:
        partitioners[label] = create_geometric_cell_partitioner(
            graph.geom_partitioner_parmetis(method, 1.02, [1, 0, 5]), ghost_mode, comm, x
        )

if has_ptscotch:
    partitioners["PT-SCOTCH"] = create_cell_partitioner(graph.partitioner_scotch(), ghost_mode, 2)

if has_kahip:
    partitioners["KaHIP"] = create_cell_partitioner(graph.partitioner_kahip(), ghost_mode, 2)

# The space-filling curve partitioner is built into DOLFINx, so it is
# always available
partitioners["DOLFINx SFC"] = create_geometric_cell_partitioner(
    graph.geom_partitioner_sfc(), ghost_mode, comm, x
)

if comm.rank == 0:
    print(f"Mesh: {6 * n**3} tetrahedra on {comm.size} rank(s)")
    print(f"{'Partitioner':<20}{'cells/rank':>12}{'imbalance':>12}{'edge cut':>12}")

for name, partitioner in partitioners.items():
    msh = create_mesh(comm, cells, cmap, x, partitioner=partitioner)
    imbalance, cut = partition_quality(msh)
    num_cells = msh.topology.index_map(msh.topology.dim).size_global
    if comm.rank == 0:
        print(f"{name:<20}{num_cells // comm.size:>12}{imbalance:>12.3f}{cut:>12}")
# -

# On a single rank there is nothing to partition: the imbalance is 1 and
# the edge cut is zero. Run the demo with, for example
#
# ```bash
# mpirun -n 12 python3 demo_partition.py
# ```
#
# to compare the partitioners. The graph partitioners cut the fewest
# facets, and the curve-based `Geom` and the DOLFINx space-filling curve
# cut roughly 20-45% more, in exchange for being much cheaper to compute --
# a trade that is worth making when the partitioning cost itself dominates,
# since graph partitioning becomes more expensive as the number of ranks
# grows while the curve partitioners barely do. All of them give an
# imbalance within a couple of percent of perfect, with the curve
# partitioners slightly ahead as they divide the cells by count.
#
# Among the graph partitioners, PT-SCOTCH typically achieves the smallest
# cut, at the cost of a slightly larger imbalance and of being
# substantially slower than ParMETIS. Its result varies a little between
# runs, so the numbers it reports are not reproducible in the way the
# others are.
#
# The cut of a curve-based partition depends on how well the curve happens
# to align with the mesh and the number of parts. Running this demo on 8
# ranks is an instructive special case: a Morton curve through a cube
# divides naturally into 8 octants, so the curve partitioners produce
# nearly optimal cube-shaped parts and cut *fewer* facets than the graph
# partitioners. Rank counts that are not a power of eight, such as the 12
# above, are more representative.
