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
#   few edges as possible. A graph partitioning function can be passed to
#   {py:func}`create_mesh <dolfinx.mesh.create_mesh>` directly, as its
#   `partitioner` argument. Three are available, depending on how DOLFINx
#   was built: ParMETIS `Kway` (multilevel k-way partitioning), PT-SCOTCH
#   and KaHIP.
# - **Geometric partitioners** work on the *positions* of the cells. Cells
#   are ordered along a space-filling curve through the mesh and the curve
#   is cut into equal pieces. This is much cheaper than graph
#   partitioning, and the cost barely grows with the number of ranks, but
#   more edges are cut. A partitioner that uses only the cell positions is
#   used via {py:func}`create_geometric_cell_partitioner
#   <dolfinx.mesh.create_geometric_cell_partitioner>`, which computes a
#   coordinate for each cell (the mean of its vertex positions) and passes
#   it to the partitioner. A **hybrid** partitioner, one that also uses
#   the graph edges as part of the partitioning decision itself rather
#   than only to determine ghost cells, is used via
#   {py:func}`create_hybrid_cell_partitioner
#   <dolfinx.mesh.create_hybrid_cell_partitioner>` instead.
#
#   ParMETIS provides two: `GeomKway` (hybrid), which uses the
#   space-filling curve to redistribute the graph and then applies k-way
#   partitioning to it, and `Geom` (purely geometric), which uses the
#   curve alone.
#
#   DOLFINx provides curve partitioners that require no external library,
#   for two curves, both purely geometric. A **Morton** ('Z-order') curve
#   is cheap to evaluate, but jumps a long way in space whenever a high
#   bit of the key changes, so consecutive cells on the curve are not
#   always neighbours. A **Hilbert** curve has no such jumps: successive
#   points on it are always neighbours, which gives more compact
#   partitions and a smaller edge cut.
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
import os
import time
from collections.abc import Callable

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from dolfinx import common, graph, has_kahip, has_parmetis, has_ptscotch
from dolfinx.fem import coordinate_element
from dolfinx.mesh import (
    CellType,
    GhostMode,
    Mesh,
    compute_cell_centroids,
    create_geometric_cell_partitioner,
    create_hybrid_cell_partitioner,
    create_mesh,
)

# -

# ## Distributed input data
#
# Mesh data is often read from file in parallel, with each rank holding an
# arbitrary block of the cells and of the geometry. Here the same
# situation is created without a file: each rank builds the block of a
# structured cube mesh of tetrahedra that corresponds to its slice of the
# global cell range.
#
# The two returned arrays describe the mesh in different index spaces, and
# understanding how they relate is the key to the input format:
#
# - `cells` holds, for each local cell, the **global** indices of its four
#   vertices. 'Global' means an index into the global list of points, and
#   the same point has the same index on every rank. Each cell appears on
#   exactly one rank, so the local blocks together list every cell once.
# - `x` holds the coordinates of the points that *this* rank supplies. The
#   global index of a point is not stored: it is implied by position,
#   being the row index plus the number of points held by all lower
#   ranks. The local blocks together supply every point once.
#
# The two distributions are **independent**. Row `j` of `x` is not the
# position of vertex `j` of the local cells, and a rank's cells will
# generally refer to vertices whose coordinates are held by other ranks.
# {py:func}`create_mesh <dolfinx.mesh.create_mesh>` resolves this: after
# the partitioner has decided cell ownership, each rank fetches the
# coordinates of the vertices it needs from whichever rank holds them. The
# same applies to the geometric partitioners, which must fetch coordinates
# before they can compute cell positions.


def cube_block(comm: MPI.Comm, n: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Local block of the cells and points of a unit cube of tetrahedra.

    Each hexahedral cell of an ``n x n x n`` grid is split into six
    tetrahedra. Cells and points are shared out over the ranks of
    ``comm`` in contiguous blocks, independently of each other.

    Args:
        comm: MPI communicator to distribute the mesh data over.
        n: Number of divisions in each direction.

    Returns:
        Tuple of ``(cells, x)``, where ``cells`` has shape
        ``(num_cells, 4)`` and row ``i`` holds the global indices of the
        four vertices of local cell ``i``, and ``x`` has shape
        ``(num_points, 3)`` and row ``j`` holds the coordinates of the
        point with global index ``j + offset``, where ``offset`` is the
        number of points held by all lower ranks.

        The two are not aligned with one another: the vertex indices in
        ``cells`` refer to the global point numbering, and the
        corresponding coordinates are in general held by other ranks.
    """
    # Cells: this rank's slice of the n^3 hexahedra
    c0, c1 = common.local_range(comm.rank, n**3, comm.size)
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
    p0, p1 = common.local_range(comm.rank, (n + 1) ** 3, comm.size)
    p = np.arange(p0, p1, dtype=np.int64)
    pz, q = np.divmod(p, (n + 1) ** 2)
    py, px = np.divmod(q, n + 1)
    x = np.stack([px, py, pz], axis=1).astype(np.float64) / n

    return np.ascontiguousarray(cells), np.ascontiguousarray(x)


# ## Randomising the input distribution
#
# The blocks above are *locality preserving*: consecutive cells of the
# structured grid are numbered consecutively, so a rank's block covers a
# compact region of the cube. Input read from file may be far less
# well behaved, with cells arriving in an order unrelated to their
# position.
#
# This matters for partitioning cost. A graph partitioner coarsens the
# dual graph, and coarsening is communication-bound when a cell's
# neighbours are held by unrelated ranks. `GeomKway` exists for exactly
# this case: it uses the space-filling curve to redistribute the graph
# so that it has spatial locality, and only then applies k-way
# partitioning.
#
# The function below moves a given fraction of each rank's cells to a
# random rank, so the input distribution can be varied from fully
# locality preserving (`fraction = 0`) to fully random
# (`fraction = 1`). Only the cells are moved; the point distribution is
# untouched, which the input format allows since the two are independent.


def redistribute_cells(
    comm: MPI.Comm, cells: npt.NDArray[np.int64], fraction: float, seed: int = 1
) -> npt.NDArray[np.int64]:
    """Move a fraction of the cells on each rank to a random rank.

    The mesh described by the cells is unchanged: cells are only moved
    between ranks, so every cell still appears exactly once.

    Args:
        comm: MPI communicator holding the cells.
        cells: Local cells, with shape ``(num_cells, 4)``.
        fraction: Fraction of the local cells to move to a rank chosen
            uniformly at random, in ``[0, 1]``. Cells not selected stay
            on the calling rank.
        seed: Random number generator seed.

    Returns:
        The cells now held by the calling rank.
    """
    if fraction <= 0.0 or comm.size == 1:
        return cells

    # Choose a destination for each local cell. Each rank decides only
    # for its own cells, so a per-rank generator gives a deterministic
    # but independent choice on each.
    rng = np.random.default_rng(seed + comm.rank)
    dest = np.full(len(cells), comm.rank)
    move = rng.random(len(cells)) < fraction
    dest[move] = rng.integers(0, comm.size, np.count_nonzero(move))

    recv = comm.alltoall([cells[dest == r] for r in range(comm.size)])
    return np.ascontiguousarray(np.concatenate(recv), dtype=np.int64)


# ## Partition quality
#
# The imbalance is computed from the number of cells owned by each rank.
# The edge cut is the global number of inter-process facets: a facet is
# listed as inter-process on the rank that owns it when its two cells are
# owned by different ranks, so summing the local counts counts each cut
# facet once. This assumes exactly two cells per facet, true for the
# tetrahedral mesh used here; meshes with branching facets (T-joints, 1D
# graph structures) can have more, and are not counted by this measure.


def partition_quality(msh: Mesh) -> tuple[float, int]:
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
# The {py:func}`coordinate element <dolfinx.fem.coordinate_element>` is the
# same in each case; only the partitioner and the input cell distribution
# differ. The geometric and hybrid partitioners need cell positions to
# partition on; {py:func}`create_mesh <dolfinx.mesh.create_mesh>` computes
# these itself, as the mean of each cell's vertex positions in the
# coordinate array `x` it is given, and passes them to the partitioner.
#
# A mesh creation time is reported alongside the quality measures, since
# the cost of partitioning is the reason to prefer a cheaper partitioner.
# It is a wall-clock time for a small mesh, so treat it as indicative
# only.

# +
comm = MPI.COMM_WORLD

# Number of divisions in each direction of the cube, giving 6 n^3
# tetrahedra. The effect of the input cell distribution on partitioning
# cost is visible only on a large mesh, so `n` is large by default. It is
# reduced when there is nothing to partition (a single rank), and in CI,
# where the demo must run quickly.
_small = comm.size == 1 or "CI" in os.environ or "GITHUB_ACTIONS" in os.environ
n = 24 if _small else 128

cells0, x = cube_block(comm, n)
cmap = coordinate_element(CellType.tetrahedron, 1)
ghost_mode = GhostMode.none

partitioners = {}

if has_parmetis:
    partitioners["ParMETIS Kway"] = graph.partitioner_parmetis()
    partitioners["ParMETIS GeomKway"] = create_hybrid_cell_partitioner(
        graph.geom_partitioner_parmetis_kway(1.02, [1, 0, 5])
    )
    partitioners["ParMETIS Geom"] = create_geometric_cell_partitioner(
        graph.geom_partitioner_parmetis()
    )

if has_ptscotch:
    partitioners["PT-SCOTCH"] = graph.partitioner_scotch()

if has_kahip:
    partitioners["KaHIP"] = graph.partitioner_kahip()

# The space-filling curve partitioners are built into DOLFINx, so they are
# always available
for label, curve in [
    ("SFC Morton", graph.SFCCurve.morton),
    ("SFC Hilbert", graph.SFCCurve.hilbert),
]:
    partitioners[label] = create_geometric_cell_partitioner(graph.geom_partitioner_sfc(curve))

if comm.rank == 0:
    print(f"Mesh: {6 * n**3} tetrahedra on {comm.size} rank(s)")

# Repeat the comparison for input distributions ranging from fully
# locality preserving to fully random
for fraction in (0.0, 0.5, 1.0):
    cells = redistribute_cells(comm, cells0, fraction)
    if comm.rank == 0:
        print(f"\nFraction of cells moved to a random rank: {fraction}")
        print(f"{'Partitioner':<20}{'imbalance':>12}{'edge cut':>12}{'time (s)':>12}")

    for name, partitioner in partitioners.items():
        comm.Barrier()
        t = time.perf_counter()
        msh = create_mesh(comm, cells, cmap, x, partitioner=partitioner, ghost_mode=ghost_mode)
        comm.Barrier()
        elapsed = comm.allreduce(time.perf_counter() - t, MPI.MAX)

        imbalance, cut = partition_quality(msh)
        if comm.rank == 0:
            print(f"{name:<20}{imbalance:>12.3f}{cut:>12}{elapsed:>12.3f}")
# -

# ## Two-stage partitioning
#
# A graph partitioner is much more expensive on a randomly distributed
# input, because its coarsening phase has to communicate with unrelated
# ranks (see above). The space-filling curve partitioner does not care
# about the input distribution at all, which suggests a two-stage scheme:
# first partition with the curve, which cheaply gives the cells spatial
# locality, then re-partition the result with the graph partitioner, which
# now runs on a well-distributed input.
#
# This is the same idea as ParMETIS `GeomKway`, but applied around any
# graph partitioner. It is shown here with PT-SCOTCH, which is the most
# sensitive to the input distribution.
#
# The first stage only needs the redistributed *cells* to feed to the
# second stage, not a mesh, so it calls the cell partitioner directly and
# exchanges the cell-vertex rows itself, using `graph.distribute`, rather
# than going through :func:`create_mesh`. This avoids paying for
# topology, geometry and ghost cells for a mesh that would otherwise be
# discarded immediately, and `graph.distribute` scales better than an
# all-to-all over the whole communicator. Since `sfc` below does not
# ghost, every cell is sent to exactly one rank, so it still appears
# exactly once, in the same vertex numbering it started with.
#
# `sfc` is a geometric partitioner, called directly with cell centroids
# (see :func:`create_geometric_cell_partitioner`), computed here with
# :func:`compute_cell_centroids` -- the same step
# :func:`create_mesh` performs internally before calling it.


def redistribute_by_partitioner(
    comm: MPI.Comm,
    cell_type: CellType,
    cells: npt.NDArray[np.int64],
    x: npt.NDArray[np.float64],
    partitioner: Callable,
) -> npt.NDArray[np.int64]:
    """Redistribute cells to the ranks assigned by a geometric partitioner.

    Args:
        comm: MPI communicator the cells and ``x`` are distributed over.
        cell_type: Cell type of ``cells``.
        cells: Local cells, with shape ``(num_cells, num_vertices)``.
        x: Geometry ('node') coordinates, as passed to :func:`create_mesh`.
        partitioner: Geometric cell partitioning function, as returned by
            :func:`create_geometric_cell_partitioner`. Called here
            directly, with its low-level ``(comm, nparts, x)``
            signature, rather than through :func:`create_mesh`.

    Returns:
        The cells assigned to this rank, in the same vertex numbering as
        the input ``cells``.
    """
    centroid = compute_cell_centroids(comm, [cell_type], [cells.reshape(-1)], comm, x)
    dest = graph.adjacencylist(
        np.asarray(partitioner(comm, comm.size, centroid), dtype=np.int32)
    )._cpp_object
    recv, _, _, _ = graph.distribute(comm, cells, dest)  # type: ignore[arg-type]
    return recv


def scotch_partitioner_time() -> float:
    """Cumulative time spent in the SCOTCH partitioner on this rank."""
    try:
        return common.timing("Compute graph partition (SCOTCH)")[1].total_seconds()
    except RuntimeError:
        return 0.0


# +
if has_ptscotch and comm.size > 1:
    cells_random = redistribute_cells(comm, cells0, 1.0)
    scotch = partitioners["PT-SCOTCH"]
    sfc = partitioners["SFC Hilbert"]

    def timed(cells: npt.NDArray[np.int64], partitioner: Callable) -> tuple[Mesh, float, float]:
        """Create a mesh, with the elapsed and the SCOTCH time."""
        comm.Barrier()
        t, t_scotch = time.perf_counter(), scotch_partitioner_time()
        msh = create_mesh(comm, cells, cmap, x, partitioner=partitioner, ghost_mode=ghost_mode)
        comm.Barrier()
        return (
            msh,
            comm.allreduce(time.perf_counter() - t, MPI.MAX),
            comm.allreduce(scotch_partitioner_time() - t_scotch, MPI.MAX),
        )

    # One stage: SCOTCH on the randomly distributed input. This repeats
    # the fraction = 1.0 row of the sweep above, deliberately, so that
    # the comparison below stands on its own.
    msh1, t1, t1_scotch = timed(cells_random, scotch)

    # Two stages: the curve first, then SCOTCH on its output. The first
    # stage redistributes the cells alone (see redistribute_by_partitioner)
    # since a full mesh from it would be thrown away immediately.
    comm.Barrier()
    t = time.perf_counter()
    cells_sfc = redistribute_by_partitioner(comm, CellType.tetrahedron, cells_random, x, sfc)
    comm.Barrier()
    t2a = comm.allreduce(time.perf_counter() - t, MPI.MAX)
    msh2b, t2b, t2b_scotch = timed(cells_sfc, scotch)

    if comm.rank == 0:
        print("\nPartitioning a randomly distributed mesh with PT-SCOTCH")
        header = f"{'Route':<28}{'imbalance':>12}{'edge cut':>12}"
        print(header + f"{'total (s)':>12}{'SCOTCH (s)':>12}")

    imb1, cut1 = partition_quality(msh1)
    imb2, cut2 = partition_quality(msh2b)
    if comm.rank == 0:
        print(f"{'SCOTCH alone':<28}{imb1:>12.3f}{cut1:>12}{t1:>12.3f}{t1_scotch:>12.3f}")
        row = f"{'SFC, then SCOTCH':<28}{imb2:>12.3f}{cut2:>12}"
        print(row + f"{t2a + t2b:>12.3f}{t2b_scotch:>12.3f}")
        print(f"{'  stage 1 (SFC)':<28}{'':>12}{'':>12}{t2a:>12.3f}")
        print(f"{'  stage 2 (SCOTCH)':<28}{'':>12}{'':>12}{t2b:>12.3f}")
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
#
# ## Effect of the input distribution
#
# Comparing the three tables shows that partition *quality* is largely
# insensitive to how the input cells are spread over the ranks: each
# partitioner reports much the same imbalance and edge cut whether the
# input is locality preserving or fully random. This is expected, since
# every partitioner sees the same mesh either way.
#
# The *cost* is not insensitive, and this is what `GeomKway` addresses.
# With a random input distribution a cell's neighbours are held by
# unrelated ranks, so the coarsening phase of a graph partitioner has to
# communicate much more, whereas `GeomKway` first uses the space-filling
# curve to restore locality. The mesh here is small and the reported time
# covers all of mesh creation rather than partitioning alone, so only a
# hint of this is visible; the gap between `Kway` and `GeomKway` widens
# with the number of cells and of ranks. The curve-only partitioners are
# almost unaffected, as they never look at the graph.
