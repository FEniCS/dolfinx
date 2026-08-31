// Copyright (C) 2020-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <array>
#include <cstdint>
#include <functional>
#include <mpi.h>
#include <optional>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

namespace dolfinx::graph
{
/// @brief Signature of functions for computing the parallel
/// partitioning of a distributed graph, using the graph edges alone.
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] local_graph Node connectivity graph.
/// @param[in] node_weights Node weights, one entry per node in
/// `local_graph`. If `std::nullopt`, nodes are treated as having equal
/// weight.
/// @param[in] edge_weights Edge weights, one entry per edge in
/// `local_graph`. If `std::nullopt`, edges are treated as having equal
/// weight.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank(s) for each input node.
using partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&,
    std::optional<std::span<const std::int32_t>>,
    std::optional<std::span<const std::int32_t>>, bool)>;

/// @brief Signature of functions for computing the parallel
/// partitioning of a distributed graph from the positions of its nodes
/// in space alone, with no access to the graph edges.
///
/// With no graph, ghost destinations cannot be computed, so this
/// signature has no `ghosting` parameter -- a partitioner of this type
/// is never asked to ghost. See ::hybrid_partition_fn for a
/// partitioning function that has access to both node positions and
/// graph edges, and so can ghost.
///
/// @note Coordinates are always `double`, regardless of the mesh's
/// scalar type: partition quality is insensitive to position precision,
/// and `double` avoids precision loss when positions are quantised
/// into keys.
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] x Node coordinates, row-major with `gdim` columns and one
/// row per node.
/// @param[in] gdim Number of coordinate components per node.
/// @param[in] node_weights Node weights, one entry per row of `x`. If
/// `std::nullopt`, nodes are treated as having equal weight. Not every
/// ::geom_partition_fn can honour node weights; one that cannot throws
/// if given anything other than `std::nullopt`.
/// @return Destination rank(s) for each input node.
using geom_partition_fn = std::function<std::vector<int>(
    MPI_Comm, int, std::span<const double>, int,
    std::optional<std::span<const std::int32_t>>)>;

/// @brief Signature of functions for computing the parallel
/// partitioning of a distributed graph using both its edges and the
/// positions of its nodes in space.
///
/// Unlike ::geom_partition_fn, a hybrid partitioner uses the graph
/// edges in the partitioning decision itself, not just for ghosting,
/// so it always needs both inputs. ParMETIS `GeomKway`, for example,
/// redistributes nodes along a space-filling curve and then applies
/// graph partitioning to the result.
///
/// @note The coordinates are always `double`, for the same reason as
/// ::geom_partition_fn.
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] local_graph Node connectivity graph.
/// @param[in] x Node coordinates, row-major with one row per node.
/// `x.size() / local_graph.num_nodes()` gives the number of coordinate
/// components per node.
/// @param[in] node_weights Node weights, one entry per node in
/// `local_graph`. If `std::nullopt`, nodes are treated as having equal
/// weight.
/// @param[in] edge_weights Edge weights, one entry per edge in
/// `local_graph`. If `std::nullopt`, edges are treated as having equal
/// weight.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank(s) for each input node.
using hybrid_partition_fn = std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, std::span<const double>,
    std::optional<std::span<const std::int32_t>>,
    std::optional<std::span<const std::int32_t>>, bool)>;

/// @brief Any of the three partitioning function shapes that
/// mesh::create_mesh accepts: ::partition_fn, ::geom_partition_fn, or
/// ::hybrid_partition_fn.
///
/// mesh::create_mesh always has the cell topology available, so it
/// builds the dual graph itself and passes it to a ::partition_fn or
/// ::hybrid_partition_fn, neither of which has any other way to obtain
/// it. For a ::geom_partition_fn or ::hybrid_partition_fn it also
/// computes cell centroids from the vertex coordinates -- using the
/// same `(commg, x, xshape)` data it uses to build the mesh -- since
/// neither has any other way to obtain them.
using AnyPartitionFunction
    = std::variant<partition_fn, geom_partition_fn, hybrid_partition_fn>;

/// @brief Whether an ::AnyPartitionFunction holds a callable
/// partitioner.
/// @param[in] partitioner Partitioner to check.
/// @return `true` if `partitioner` holds a callable function, `false`
/// if it is default-constructed (not callable).
bool has_partitioner(const AnyPartitionFunction& partitioner);

/// @brief Partition graph across processes using the default graph
/// partitioner.
///
/// @param[in] comm MPI communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into.
/// @param[in] local_graph Node connectivity graph.
/// @param[in] node_weights Node weights. Each partition aims to have the same
/// sum of node weights. If `std::nullopt`, nodes are treated as having
/// equal weight.
/// @param[in] edge_weights Edge weights. Higher values increase the likelihood
/// that adjacent cells will be on the same partition. If `std::nullopt`,
/// edges are treated as having equal weight.
/// @param[in] ghosting Flag to enable ghosting of the output node
/// distribution.
/// @return Destination rank for each input node.
AdjacencyList<std::int32_t> partition_graph(
    MPI_Comm comm, int nparts, const AdjacencyList<std::int64_t>& local_graph,
    std::optional<std::span<const std::int32_t>> node_weights,
    std::optional<std::span<const std::int32_t>> edge_weights, bool ghosting);

/// @brief An ::AnyPartitionFunction together with the node weights it
/// should be called with, if any.
///
/// Bundles the two arguments that are otherwise threaded through
/// mesh::create_mesh and mesh::impl::partition_cells separately, since
/// a weight only makes sense alongside the partitioner it is meant
/// for.
struct Partitioner
{
  /// Partitioning function. Defaults to ::partition_graph, the default
  /// graph partitioner.
  AnyPartitionFunction fn = partition_fn(partition_graph);

  /// Node weights, one entry per node the partitioner is called with.
  /// If `std::nullopt`, nodes are treated as having equal weight.
  std::optional<std::span<const std::int32_t>> node_weights = std::nullopt;
};

/// Tools for distributed graphs
///
/// @todo Add a function that sends data to the 'owner'
namespace build
{
/// @brief Distribute adjacency list nodes to destination ranks.
///
/// The global index of the `i`th node (row) in `list` is assumed to be
/// `i` plus the offset for this rank, i.e. the number of nodes owned by
/// lower-ranked processes.
///
/// @note Collective.
///
/// @note The neighbourhood communicator is built with
/// MPI::compute_graph_edges_nbx, which discovers incoming edges from
/// the outgoing edges alone via the scalable NBX consensus algorithm,
/// keeping the communication pattern sparse. This is not free: it
/// costs one or more non-blocking consensus rounds, so calling this
/// function repeatedly (e.g. once per cell type) has a real cost.
///
/// @param[in] comm MPI Communicator that `list`/`destinations` are
/// distributed across.
/// @param[in] list The adjacency list to distribute.
/// @param[in] destinations Destination rank(s) for the `i`th node in
/// `list`. The first rank is the 'owner' of the node; any further ranks
/// receive it as a ghost.
/// @return
/// 1. Received adjacency list for this process. Nodes owned by this
///    process come first, followed by any ghost nodes.
/// 2. Source rank of each node in (1), i.e. the rank it was sent from.
/// 3. Original global index of each node in (1).
/// 4. Owning rank of the ghost nodes among (1). This has one entry per
///    ghost node -- the trailing entries of (1) -- not one entry per
///    node of (1).
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
distribute(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& list,
           const graph::AdjacencyList<std::int32_t>& destinations);

/// @brief Distribute rows of a fixed-degree array to destination ranks.
///
/// The global index of the `i`th row of `list` is assumed to be `i`
/// plus the offset for this rank, i.e. the number of rows owned by
/// lower-ranked processes.
///
/// @note Collective.
///
/// @note The neighbourhood communicator is built with
/// MPI::compute_graph_edges_nbx, which discovers incoming edges from
/// the outgoing edges alone via the scalable NBX consensus algorithm,
/// keeping the communication pattern sparse. This is not free: it
/// costs one or more non-blocking consensus rounds, so calling this
/// function repeatedly (e.g. once per cell type) has a real cost.
///
/// @param[in] comm MPI Communicator that `list`/`destinations` are
/// distributed across.
/// @param[in] list Constant degree (valency) data, flattened row-major
/// with shape `shape`.
/// @param[in] shape Shape `(num_nodes, degree)` of `list`.
/// @param[in] destinations Destination rank(s) for the `i`th row of
/// `list`. The first rank is the 'owner' of the row; any further ranks
/// receive it as a ghost.
/// @return
/// 1. Received rows for this process, flattened row-major with shape
///    (num_nodes, degree). Rows owned by this process come first,
///    followed by any ghost rows.
/// 2. Source rank of each row in (1), i.e. the rank it was sent from.
/// 3. Original global index of each row in (1).
/// 4. Owning rank of the ghost rows among (1). This has one entry per
///    ghost row -- the trailing rows of (1) -- not one entry per row
///    of (1).
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
