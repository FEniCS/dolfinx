// Copyright (C) 2020-2026 Garth N. Wells and Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "partition.h"
#include <array>
#include <optional>
#include <span>

namespace dolfinx::graph
{

/// @todo Is it un-documented that the owning rank must come first in
/// reach list of edges?
///
/// @param[in] comm The communicator
/// @param[in] graph Graph, using global indices for graph edges
/// @param[in] node_disp The distribution of graph nodes across MPI
/// ranks. The global index `gidx` of local index `lidx` is `lidx +
/// node_disp[my_rank]`.
/// @param[in] part The destination rank for owned nodes, i.e. `dest[i]`
/// is the destination of the node with local index `i`.
/// @return Destination ranks for each local node.
template <typename T>
graph::AdjacencyList<int> compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<T>& node_disp, const std::vector<T>& part);

namespace scotch
{
#ifdef HAS_PTSCOTCH
/// @brief PT-SCOTCH partitioning strategies.
///
/// See PT-SCOTCH documentation for details.
enum class strategy : std::uint8_t
{
  ///< SCOTCH's own default strategy, which is not the DOLFINx default
  ///< (see ::partitioner)
  none,
  balance,
  quality,
  safety,
  speed,
  scalability
};

/// @brief Create a graph partitioning function that uses PT-SCOTCH.
///
/// @note The default strategy is strategy::speed rather than
/// strategy::none (SCOTCH's own default). For a mesh dual graph,
/// strategy::speed has been measured to be around 20% faster with no
/// measurable change in the number of cut edges. Note also that
/// strategy::quality is slower *and* cuts more edges than
/// strategy::none for such graphs.
///
/// @param[in] strategy The SCOTCH strategy
/// @param[in] imbalance The allowable imbalance (between 0 and 1). The
/// smaller value the more balanced the partitioning must be. Note that
/// this does not affect the run time appreciably.
/// @param[in] seed Random number generator seed
/// @return A graph partitioning function
graph::partition_fn partitioner(scotch::strategy strategy = strategy::speed,
                                double imbalance = 0.025, int seed = 0);
#endif

} // namespace scotch

namespace parmetis
{
#ifdef HAS_PARMETIS
/// @brief Create a graph partitioning function that uses ParMETIS.
///
/// @note ParMETIS fails (crashes) if an MPI rank has no part of the
/// graph. If necessary, the communicator should be split to avoid this
/// situation.
///
/// @param[in] imbalance Imbalance tolerance. See ParMETIS manual for
/// details
/// (https://github.com/KarypisLab/ParMETIS/blob/main/manual/manual.pdf).
/// @param[in] options The ParMETIS option. See ParMETIS manual for
/// details.
/// @return A graph partitioning function.
graph::partition_fn partitioner(double imbalance = 1.02,
                                std::array<int, 3> options = {1, 0, 5});

/// @brief Create a graph re-partitioning function that uses ParMETIS.
///
/// Unlike ::partitioner, which computes a partition from scratch, the
/// returned function treats the graph's *current* distribution as the
/// current partition -- the nodes held by a rank are taken to be
/// assigned to that rank -- and computes a new partition that balances
/// the load while limiting how many nodes have to move. This is
/// appropriate when a distributed mesh needs re-balancing, e.g. after
/// non-uniform refinement, where re-partitioning from scratch would
/// migrate almost every cell.
///
/// @note ParMETIS fails (crashes) if an MPI rank has no part of the
/// graph. If necessary, the communicator should be split to avoid this
/// situation.
///
/// @note The number of parts must equal the size of the communicator, as
/// the current partition is taken from the data distribution.
///
/// @param[in] ipc2redist Ratio of the cost of inter-process
/// communication (edge cut) to the cost of moving a node between ranks.
/// A small value (e.g. 0.001) prioritises leaving nodes where they are,
/// and a large value (e.g. 1000) prioritises the quality of the new
/// partition. See ParMETIS manual for details
/// (https://github.com/KarypisLab/ParMETIS/blob/main/manual/manual.pdf).
/// @param[in] imbalance Imbalance tolerance.
/// @param[in] options The ParMETIS options. See ParMETIS manual for
/// details.
/// @return A graph re-partitioning function.
graph::partition_fn repartitioner(double ipc2redist = 1000.0,
                                  double imbalance = 1.02,
                                  std::array<int, 3> options = {1, 0, 5});

/// @brief A geometric graph partitioning function (matching
/// ::geom_partition_fn) that uses ParMETIS to order the coordinates
/// along a space-filling curve (`ParMETIS_V3_PartGeom`).
///
/// Unlike ::repartitioner and ::geom_partitioner_kway, this is not a
/// factory: it takes no configuration, so it is the partitioning
/// function itself, ready to use directly.
///
/// The graph edges are unused for partitioning, so this is much cheaper
/// than ::geom_partitioner_kway, but cuts more edges.
///
/// @note ParMETIS fails (crashes) if an MPI rank has no graph nodes. This
/// function partitions only on ranks with nodes, mapping output part
/// indices back to ranks in `comm`.
///
/// @note This partitions into one part per rank with graph nodes.
/// `nparts` must equal the communicator size.
///
/// @note ::geom_partition_fn has no graph, so this has no `ghosting`
/// parameter and is never asked to ghost.
///
/// @param[in] comm MPI Communicator that the graph is distributed
/// across.
/// @param[in] nparts Number of partitions to divide graph nodes into;
/// must equal the size of `comm`.
/// @param[in] x Node coordinates, row-major with `gdim` columns and one
/// row per node.
/// @param[in] gdim Number of coordinate components per node.
/// @param[in] node_weights Unused: `ParMETIS_V3_PartGeom` does not
/// support node weights. Must be `std::nullopt`.
/// @return Destination rank for each input node.
std::vector<int>
geom_partitioner(MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
                 std::optional<std::span<const std::int32_t>> node_weights);

/// @brief Create a geometric graph partitioning function that uses
/// ParMETIS to order the coordinates along a space-filling curve,
/// redistribute the graph accordingly, then apply multilevel k-way
/// partitioning to the redistributed graph (`ParMETIS_V3_PartGeomKway`).
///
/// The partition quality is comparable to ::partitioner, and the
/// space-filling curve redistribution makes the k-way phase markedly
/// cheaper when the input graph distribution does not reflect the node
/// positions.
///
/// @note ParMETIS fails (crashes) if an MPI rank has no part of the
/// graph. If necessary, the communicator should be split to avoid this
/// situation.
///
/// @param[in] imbalance Imbalance tolerance. See ParMETIS manual for
/// details
/// (https://github.com/KarypisLab/ParMETIS/blob/main/manual/manual.pdf).
/// @param[in] options The ParMETIS options. See ParMETIS manual for
/// details.
/// @return A hybrid graph partitioning function. It requires both `x`
/// and `local_graph`, since the graph edges are used as well as the
/// coordinates. Node and edge weights, if supplied at call time, are
/// passed on to `ParMETIS_V3_PartGeomKway`.
graph::hybrid_partition_fn geom_partitioner_kway(double imbalance = 1.02,
                                                 std::array<int, 3> options
                                                 = {1, 0, 5});
#endif
} // namespace parmetis

/// Interfaces to KaHIP parallel partitioner
namespace kahip
{
#ifdef HAS_KAHIP
/// @brief Create a graph partitioning function that uses KaHIP.
///
/// @param[in] mode The KaHiP partitioning mode (see
/// https://github.com/KaHIP/KaHIP/blob/master/parallel/parallel_src/interface/parhip_interface.h)
/// @param[in] seed The KaHiP random number generator seed
/// @param[in] imbalance The allowable imbalance
/// @param[in] suppress_output Suppresses KaHIP output if true
/// @return A KaHIP graph partitioning function with specified parameter
/// options
graph::partition_fn partitioner(int mode = 1, int seed = 1,
                                double imbalance = 0.03,
                                bool suppress_output = true);
#endif
} // namespace kahip

} // namespace dolfinx::graph
