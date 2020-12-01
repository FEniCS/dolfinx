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

// Developer note: ptscotch.h is not part of the public interface,
// therefore this header (SCOTCH.h) should not be placed in the public
// interface of DOLFIN.
extern "C"
{
#include <ptscotch.h>
}

/// Interface to SCOTCH-PT (parallel version)
namespace dolfinx::graph::SCOTCH
{

/// Compute distributed graph partition
/// @param mpi_comm MPI Communicator
/// @param nparts Number of partitions to divide graph nodes into
/// @param local_graph Node connectivity graph
/// @param node_weights Weight for each node (optional)
/// @param num_ghost_nodes Number of graph nodes which are owned on
///   other processes
/// @param ghosting Flag to enable ghosting of the output node
///   distribution
/// @return Destination rank for each input node
AdjacencyList<std::int32_t>
partition(const MPI_Comm mpi_comm, const int nparts,
          const AdjacencyList<SCOTCH_Num>& local_graph,
          const std::vector<std::size_t>& node_weights,
          std::int32_t num_ghost_nodes, bool ghosting);

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

} // namespace dolfinx::graph::SCOTCH
