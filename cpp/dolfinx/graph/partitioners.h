// Copyright (C) 2020-2026 Garth N. Wells and Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "partition.h"
#include <array>

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
  ///< SCOTCH default strategy
  none,
  balance,
  quality,
  safety,
  speed,
  scalability
};

/// @brief Create a graph partitioning function that uses PT-SCOTCH.
///
/// @param[in] strategy The SCOTCH strategy
/// @param[in] imbalance The allowable imbalance (between 0 and 1). The
/// smaller value the more balanced the partitioning must be.
/// @param[in] seed Random number generator seed
/// @return A graph partitioning function
graph::partition_fn partitioner(scotch::strategy strategy = strategy::none,
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
graph::partition_fn partitioner(double imbalance = 1.02,
                                std::array<int, 3> options = {1, 0, 5});

/// @brief ParMETIS partitioning methods that use node coordinates.
enum class geom_method : std::uint8_t
{
  /// Space-filling curve ordering of the coordinates, used to
  /// redistribute the graph before multilevel k-way partitioning
  /// (`ParMETIS_V3_PartGeomKway`). The partition quality is comparable
  /// to ::partitioner, and the redistribution makes the k-way phase
  /// markedly cheaper when the input graph distribution does not
  /// reflect the node positions.
  kway,

  /// Space-filling curve ordering of the coordinates only, with the
  /// graph edges unused (`ParMETIS_V3_PartGeom`). Much cheaper than
  /// ::kway, but cuts more edges.
  curve
};

/// @brief Create a geometric graph partitioning function that uses
/// ParMETIS.
///
/// @note ParMETIS fails (crashes) if an MPI rank has no part of the
/// graph. If necessary, the communicator should be split to avoid this
/// situation.
///
/// @note `geom_method::curve` partitions into one part per rank of the
/// communicator, so `nparts` must equal the communicator size.
///
/// @param[in] method Partitioning method.
/// @param[in] imbalance Imbalance tolerance (`geom_method::kway` only).
/// See ParMETIS manual for details
/// (https://github.com/KarypisLab/ParMETIS/blob/main/manual/manual.pdf).
/// @param[in] options The ParMETIS options (`geom_method::kway` only).
/// See ParMETIS manual for details.
/// @return A geometric graph partitioning function.
graph::geom_partition_fn
geom_partitioner(geom_method method = geom_method::kway,
                 double imbalance = 1.02,
                 std::array<int, 3> options = {1, 0, 5});
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
