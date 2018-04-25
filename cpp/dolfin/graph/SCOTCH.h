// Copyright (C) 2010-2013 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <cstddef>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif


namespace dolfin
{
// Forward declarations
namespace mesh
{
class CellType;
}

namespace graph
{

template <typename T>
class CSRGraph;

/// This class provides an interface to SCOTCH-PT (parallel version)

class SCOTCH
{
public:
  // Compute cell partitions from distributed dual graph. Returns
  // (partition, ghost_proc)
  static std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
  partition(const MPI_Comm mpi_comm, const CSRGraph<SCOTCH_Num>& local_graph,
            const std::vector<std::size_t>& node_weights,
            std::int32_t num_ghost_nodes);

  /// Compute reordering (map[old] -> new) using
  /// Gibbs-Poole-Stockmeyer (GPS) re-ordering
  /// @param graph (Graph)
  ///   Input graph
  /// @param num_passes (std::size_t)
  ///   Number of passes to use in GPS algorithm
  /// @return std::vector<int>
  ///   Mapping from old to new nodes
  /// @return std::vector<int>
  ///   Mapping from new to old nodes (inverse map)
  static std::pair<std::vector<int>, std::vector<int>>
  compute_gps(const Graph& graph, std::size_t num_passes = 5);

  /// Compute graph re-ordering
  /// @param graph (Graph)
  ///   Input graph
  /// @param scotch_strategy (string)
  ///   SCOTCH parameters
  /// @return std::vector<int>
  ///   Mapping from old to new nodes
  /// @return std::vector<int>
  ///   Mapping from new to old nodes (inverse map)
  static std::pair<std::vector<int>, std::vector<int>>
  compute_reordering(const Graph& graph, std::string scotch_strategy = "");
};
} // namespace graph
} // namespace dolfin
