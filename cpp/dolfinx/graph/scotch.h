// Copyright (C) 2010-2020 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include "partition.h"
#include <cstdint>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

/// Interface to SCOTCH-PT
namespace dolfinx::graph::scotch
{

/// SCOTCH partitioning strategies
enum class strategy
{
  none, // SCOTCH default strategy
  balance,
  quality,
  safety,
  speed,
  scalability
};

/// Create a graph partitioning function that uses SCOTCH
///
/// @param[in] strategy The SCOTCH strategy
/// @param[in] imbalance The allowable imbalance (between 0 and 1). The
/// smaller value the more balanced the partitioning must be.
/// @param[in] seed Random number generator seed
/// @return A graph partitioning function
graph::partition_fn partitioner(scotch::strategy strategy = strategy::none,
                                double imbalance = 0.025, int seed = 0);

/// Compute reordering (map[old] -> new) using Gibbs-Poole-Stockmeyer
/// (GPS) re-ordering
///
/// @param[in] graph Input graph
/// @param[in] num_passes Number of passes to use in GPS algorithm
/// @return (map from old to new nodes, map from new to old nodes
/// (inverse map))
std::pair<std::vector<int>, std::vector<int>>
compute_gps(const AdjacencyList<std::int32_t>& graph,
            std::size_t num_passes = 5);

/// Compute graph re-ordering
///
/// @param[in] graph Input graph
/// @param[in] scotch_strategy (string) SCOTCH parameters
/// @return (map from old to new nodes, map from new to old nodes
/// (inverse map))
std::pair<std::vector<int>, std::vector<int>>
compute_reordering(const AdjacencyList<std::int32_t>& graph,
                   std::string scotch_strategy = "");

} // namespace dolfinx::graph::scotch
