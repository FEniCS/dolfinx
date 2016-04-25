// Copyright (C) 2010-2013 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-02-10
// Last changed: 2014-01-09

#ifndef __SCOTCH_PARTITIONER_H
#define __SCOTCH_PARTITIONER_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include "Graph.h"

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
    static void compute_partition(
      const MPI_Comm mpi_comm,
      std::vector<int>& cell_partition,
      std::map<std::int64_t, std::vector<int>>& ghost_procs,
      const boost::multi_array<std::int64_t, 2>& cell_vertices,
      const std::vector<std::int64_t>& global_cell_indices,
      const std::vector<std::size_t>& cell_weight,
      const std::int64_t num_global_vertices,
      const std::int64_t num_global_cells,
      const CellType& cell_type);

    /// Compute reordering (map[old] -> new) using
    /// Gibbs-Poole-Stockmeyer re-ordering
    static std::vector<int> compute_gps(const Graph& graph,
                                        std::size_t num_passes=5);

    // Compute graph re-ordering
    static std::vector<int>
      compute_reordering(const Graph& graph,
                         std::string scotch_strategy="");

    // Compute graph re-ordering
    static
      void compute_reordering(const Graph& graph,
                              std::vector<int>& permutation,
                              std::vector<int>& inverse_permutation,
                              std::string scotch_strategy="");

  private:

    // Compute cell partitions from distributed dual graph
    static void partition(
      const MPI_Comm mpi_comm,
      const std::vector<std::set<std::size_t>>& local_graph,
      const std::vector<std::size_t>& node_weights,
      const std::set<std::size_t>& ghost_vertices,
      const std::vector<std::int64_t>& global_cell_indices,
      const std::size_t num_global_vertices,
      std::vector<int>& cell_partition,
      std::map<std::int64_t, std::vector<int>>& ghost_procs);

  };

}

#endif
