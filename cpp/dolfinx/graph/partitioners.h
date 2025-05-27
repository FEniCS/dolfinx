// Copyright (C) 2020-2023 Garth N. Wells and Igor A. Baratta
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
enum class strategy
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
