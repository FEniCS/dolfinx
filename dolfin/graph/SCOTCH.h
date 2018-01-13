// Copyright (C) 2010-2013 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "Graph.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>

#include "CSRGraph.h"

namespace dolfin
{
// Forward declarations
class CellType;
class LocalMeshData;

/// This class provides an interface to SCOTCH-PT (parallel version)

class SCOTCH
{
public:
  /// Compute cell partition from local mesh data.  The vector
  /// cell_partition contains the desired destination process
  /// numbers for each cell.  Cells shared on multiple processes
  /// have an entry in ghost_procs pointing to the set of sharing
  /// process numbers.
  /// @param mpi_comm (MPI_Comm)
  /// @param cell_partition (std::vector<int>)
  /// @param ghost_procs (std::map<std::int64_t, std::vector<int>>)
  /// @param cell_vertices (const boost::multi_array<std::int64_t, 2>)
  /// @param cell_weight (const std::vector<std::size_t>)
  /// @param num_global_vertices (const std::int64_t)
  /// @param num_global_cells (const std::int64_t)
  /// @param cell_type (const CellType)
  ///
  static void
  compute_partition(const MPI_Comm mpi_comm, std::vector<int>& cell_partition,
                    std::map<std::int64_t, std::vector<int>>& ghost_procs,
                    const boost::multi_array<std::int64_t, 2>& cell_vertices,
                    const std::vector<std::size_t>& cell_weight,
                    const std::int64_t num_global_vertices,
                    const std::int64_t num_global_cells,
                    const CellType& cell_type);

  /// Compute reordering (map[old] -> new) using
  /// Gibbs-Poole-Stockmeyer (GPS) re-ordering
  /// @param graph (Graph)
  ///   Input graph
  /// @param num_passes (std::size_t)
  ///   Number of passes to use in GPS algorithm
  /// @return std::vector<int>
  ///   Mapping from old to new nodes
  static std::vector<int> compute_gps(const Graph& graph,
                                      std::size_t num_passes = 5);

  /// Compute graph re-ordering
  /// @param graph (Graph)
  ///   Input graph
  /// @param scotch_strategy (string)
  ///   SCOTCH parameters
  /// @return std::vector<int>
  ///   Mapping from old to new nodes
  static std::vector<int> compute_reordering(const Graph& graph,
                                             std::string scotch_strategy = "");

  /// Compute graph re-ordering
  /// @param graph (Graph)
  /// @param permutation (std::vector<int>)
  /// @param inverse_permutation (std::vector<int>)
  /// @param scotch_strategy (std::string)
  static void compute_reordering(const Graph& graph,
                                 std::vector<int>& permutation,
                                 std::vector<int>& inverse_permutation,
                                 std::string scotch_strategy = "");

private:
  // Compute cell partitions from distributed dual graph. Note that
  // local_graph is not const since we share the data with SCOTCH,
  // and the SCOTCH interface is not const-correct.
  template <typename T>
  static void partition(const MPI_Comm mpi_comm, CSRGraph<T>& local_graph,
                        const std::vector<std::size_t>& node_weights,
                        const std::set<std::int64_t>& ghost_vertices,
                        const std::size_t num_global_vertices,
                        std::vector<int>& cell_partition,
                        std::map<std::int64_t, std::vector<int>>& ghost_procs);
};
}


