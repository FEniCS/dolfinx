// Copyright (C) 2020-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <mpi.h>
#include <span>
#include <utility>
#include <vector>

#include <iostream>

namespace dolfinx::graph
{
/// @brief Signature of functions for computing the parallel
/// partitioning of a distributed graph.
/// @param[in] comm MPI Communicator that the graph is distributed
/// across
/// @param[in] nparts Number of partitions to divide graph nodes into
/// @param[in] local_graph Node connectivity graph
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution
/// @return Destination rank for each input node
using partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, bool)>;

/// @brief Signature of functions for computing the parallel partitioning
/// of a distributed graph whose nodes have a position in space.
///
/// As ::partition_fn, with the addition of a coordinate for each node of
/// the graph. Partitioners of this form can use the node positions
/// instead of, or in addition to, the graph edges.
///
/// @note The coordinates are always `double`, whatever the scalar type of
/// the data they were derived from. Which nodes end up in which part is
/// not sensitive to the precision of the positions, so there is nothing
/// to be gained from partitioning single precision positions in single
/// precision, and computing the keys in `double` removes any question of
/// precision loss when positions are quantised. A caller that holds
/// single precision coordinates and wants to avoid the conversion can use
/// ::partition_sfc_morton or ::partition_sfc_hilbert directly, which are
/// templated on the scalar type.
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] local_graph Node connectivity graph.
/// @param[in] x Node coordinates, row-major with `gdim` columns and one
/// row per node of `local_graph`.
/// @param[in] gdim Number of coordinate components per node.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank(s) for each input node.
using geom_partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, std::span<const double>,
    int, bool)>;

/// @brief Partition graph across processes using the default graph
/// partitioner.
///
/// @param[in] comm MPI communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] local_graph Node connectivity graph.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank for each input node.
AdjacencyList<std::int32_t>
partition_graph(MPI_Comm comm, int nparts,
                const AdjacencyList<std::int64_t>& local_graph, bool ghosting);

/// @brief Partition points into `nparts` groups of (approximately) equal
/// size using a Morton ('Z-order') space-filling curve.
///
/// Points are ordered by the Morton key of their position in the global
/// bounding box, and the resulting order is cut into `nparts` equal
/// pieces. Splitters are selected from a gathered sample of the keys, so
/// the cost is linear in the number of local points plus one all-gather
/// of the sample.
///
/// Compared to a graph partitioner, this is much cheaper (no graph is
/// required and the cost is nearly independent of the number of ranks)
/// and gives a near-perfect load balance, at the cost of a larger number
/// of cut edges (typically tens of percent for a mesh dual graph).
///
/// A Morton curve jumps a long way in space each time a high bit of the
/// key changes, so consecutive points on the curve are not always close
/// together. ::partition_sfc_hilbert avoids this.
///
/// @note Collective.
///
/// @note The keys are computed in `double` whatever `T` is, so a single
/// and a double precision call on the same positions give the same
/// partition. `T` exists so that a caller holding single precision
/// coordinates need not convert them; it does not change the result.
/// Partitioners built on ::geom_partition_fn always pass `double`.
///
/// @param[in] comm MPI communicator that the points are distributed
/// across.
/// @param[in] nparts Number of partitions to divide the points into.
/// @param[in] local_graph Node connectivity graph, with one node per
/// point. It is used only to determine ghost nodes, i.e. it has no
/// influence on which part a point is assigned to, and is not read at
/// all when `ghosting` is false.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point. Must be
/// 1, 2 or 3.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank(s) for each point, the owning rank first.
/// @tparam T Scalar type of the coordinates.
template <std::floating_point T>
AdjacencyList<std::int32_t>
partition_sfc_morton(MPI_Comm comm, int nparts,
                     const AdjacencyList<std::int64_t>& local_graph,
                     std::span<const T> x, int gdim, bool ghosting);

/// @brief Partition points into `nparts` groups of (approximately) equal
/// size using a Hilbert space-filling curve.
///
/// As ::partition_sfc_morton, but points are ordered along a Hilbert
/// curve. Successive points on a Hilbert curve are always neighbours in
/// space, which a Morton curve does not guarantee, so the resulting
/// partitions are more compact and cut fewer edges. Computing the curve
/// index is more expensive than a Morton key, but in both cases the cost
/// is dominated by the sampling and the search for each point's part.
///
/// @note Collective.
///
/// @note As ::partition_sfc_morton, `T` only saves the caller a
/// conversion; it does not change the partition.
///
/// @param[in] comm MPI communicator that the points are distributed
/// across.
/// @param[in] nparts Number of partitions to divide the points into.
/// @param[in] local_graph Node connectivity graph, with one node per
/// point. See ::partition_sfc_morton.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point. Must be
/// 1, 2 or 3.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank(s) for each point, the owning rank first.
/// @tparam T Scalar type of the coordinates.
template <std::floating_point T>
AdjacencyList<std::int32_t>
partition_sfc_hilbert(MPI_Comm comm, int nparts,
                      const AdjacencyList<std::int64_t>& local_graph,
                      std::span<const T> x, int gdim, bool ghosting);

/// Space-filling curve partitioner
namespace sfc
{
/// @brief Space-filling curves that nodes can be ordered along.
enum class curve : std::uint8_t
{
  /// Morton ('Z-order') curve, see ::partition_sfc_morton
  morton,

  /// Hilbert curve, see ::partition_sfc_hilbert
  hilbert
};

/// @brief Create a geometric partitioning function that orders nodes
/// along a space-filling curve.
///
/// The graph edges are used only to determine ghost nodes, i.e. the
/// partition itself is computed from the node coordinates alone.
///
/// @note The default is curve::hilbert. For the dual graph of a
/// tetrahedral mesh on 20 ranks, it was measured to cut 10% fewer edges
/// than curve::morton (457772 against 508750 for 12.6M cells), for a
/// similar cost.
///
/// @param[in] curve Space-filling curve to order the nodes along.
/// @return A geometric graph partitioning function.
graph::geom_partition_fn partitioner(sfc::curve curve = sfc::curve::hilbert);
} // namespace sfc

/// Tools for distributed graphs
///
/// @todo Add a function that sends data to the 'owner'
namespace build
{
/// @brief Distribute adjacency list nodes to destination ranks.
///
/// The global index of each node is assumed to be the local index plus
/// the offset for this rank.
///
/// @param[in] comm MPI Communicator
/// @param[in] list The adjacency list to distribute
/// @param[in] destinations Destination ranks for the ith node in the
/// adjacency list. The first rank is the 'owner' of the node.
/// @return
/// 1. Received adjacency list for this process
/// 2. Source ranks for each node in the adjacency list
/// 3. Original global index for each node in the adjacency list
/// 4. Owning rank of ghost nodes.
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
distribute(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
           const graph::AdjacencyList<std::int32_t>& destinations);

/// @brief Distribute fixed size nodes to destination ranks.
///
/// The global index of each node is assumed to be the local index plus
/// the offset for this rank.
///
/// @param[in] comm MPI Communicator
/// @param[in] list Constant degree (valency) adjacency list. The array
/// shape is (num_nodes, degree). Storage is row-major.
/// @param[in] shape Shape `(num_nodes, degree)` of `list`.
/// @param[in] destinations Destination ranks for the ith node (row) of
/// `list`. The first rank is the 'owner' of the node.
/// @return
/// 1. Received adjacency list on this process. The array shape is
/// (num_nodes, degree). Storage is row-major.
/// 2. Source rank for each received node.
/// 3. Original global index for each received node.
/// 4. Owning rank of ghost nodes.
std::tuple<std::vector<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
distribute(MPI_Comm comm, std::span<const std::int64_t> list,
           std::array<std::size_t, 2> shape,
           const graph::AdjacencyList<std::int32_t>& destinations);

/// @brief Take a set of distributed input global indices, including
/// ghosts, and determine the new global indices after remapping.
///
/// Each rank receive 'input' global indices `[i0, i1, ..., i(m-1), im,
/// ..., i(n-1)]`, where the first `m` indices are owned by the caller
/// and the remainder are 'ghosts' indices that are owned by other ranks.
///
/// Each rank assigns new global indices to its owned indices. The new
/// index is the rank offset (scan of the number of indices owned by the
/// lower rank processes, typically computed using `MPI_Exscan` with
/// `MPI_SUM`), i.e. `i1 -> offset + 1`, `i2 -> offset + 2`, etc. Ghost
/// indices are number by the remote owning processes. The function
/// returns the new ghost global indices by retrieving the new indices
/// from the owning ranks.
///
/// @param[in] comm MPI communicator
/// @param[in] owned_indices List of owned global indices. It should not
/// contain duplicates, and these indices must not appear in
/// `owned_indices` on other ranks.
/// @param[in] ghost_indices List of ghost global indices.
/// @param[in] ghost_owners The owning rank for each entry in
/// `ghost_indices`.
/// @param[in] num_threads Number of threads to use.
/// @return New global indices for the ghost indices.
std::vector<std::int64_t>
compute_ghost_indices(MPI_Comm comm,
                      std::span<const std::int64_t> owned_indices,
                      std::span<const std::int64_t> ghost_indices,
                      std::span<const int> ghost_owners, int num_threads);

/// Given an adjacency list with global, possibly non-contiguous, link
/// indices and a local adjacency list with contiguous link indices
/// starting from zero, compute a local-to-global map for the links.
/// Both adjacency lists must have the same shape.
///
/// @param[in] global Adjacency list with global link indices.
/// @param[in] local Adjacency list with local, contiguous link indices.
/// @return Map from local index to global index, which if applied to
/// the local adjacency list indices would yield the global adjacency
/// list.
std::vector<std::int64_t>
compute_local_to_global(std::span<const std::int64_t> global,
                        std::span<const std::int32_t> local);

/// @brief Compute a local0-to-local1 map from two local-to-global maps
/// with common global indices.
///
/// @param[in] local0_to_global Map from local0 indices to global
/// indices
/// @param[in] local1_to_global Map from local1 indices to global
/// indices
/// @return Map from local0 indices to local1 indices
std::vector<std::int32_t>
compute_local_to_local(std::span<const std::int64_t> local0_to_global,
                       std::span<const std::int64_t> local1_to_global);
} // namespace build

} // namespace dolfinx::graph
