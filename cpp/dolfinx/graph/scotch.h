// Copyright (C) 2010-2013 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include <cstdint>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

/// Interface to SCOTCH-PT (parallel version)
namespace dolfinx::graph::scotch
{

/// Create a graph partitioning function that uses PT-SCOTCH
///
/// @param[in] mode The KaHiP partitioning mode (see
/// https://github.com/KaHIP/KaHIP/blob/master/parallel/parallel_src/interface/parhip_interface.h)
/// @param[in] seed The KaHiP random number generator seed
/// @param[in] imbalance The allowable imbalance
/// @param[in] suppress_output Suppresses KaHIP output if true
/// @return A SCOTCH graph partitioning function
std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, std::int32_t, bool)>
partitioner();

// /// Compute partitioning of a distributed graph
// ///
// /// @param mpi_comm MPI Communicator
// /// @param nparts Number of partitions to divide graph nodes into
// /// @param local_graph Node connectivity graph
// /// @param num_ghost_nodes Number of graph nodes which are owned on
// /// other processes
// /// @param ghosting Flag to enable ghosting of the output node
// /// distribution
// /// @return Destination rank for each input node
// AdjacencyList<std::int32_t>
// partition(const MPI_Comm mpi_comm, int nparts,
//           const AdjacencyList<std::int64_t>& local_graph,
//           std::int32_t num_ghost_nodes, bool ghosting);

/// Compute reordering (map[old] -> new) using Gibbs-Poole-Stockmeyer
/// (GPS) re-ordering
/// @param[in] graph Input graph
/// @param[in] num_passes Number of passes to use in GPS algorithm
/// @return (map from old to new nodes, map from new to old nodes
///   (inverse map))
std::pair<std::vector<int>, std::vector<int>>
compute_gps(const AdjacencyList<std::int32_t>& graph,
            std::size_t num_passes = 5);

/// Compute graph re-ordering
/// @param[in] graph Input graph
/// @param[in] scotch_strategy (string) SCOTCH parameters
/// @return (map from old to new nodes, map from new to old nodes
///   (inverse map))
std::pair<std::vector<int>, std::vector<int>>
compute_reordering(const AdjacencyList<std::int32_t>& graph,
                   std::string scotch_strategy = "");

} // namespace dolfinx::graph::scotch
