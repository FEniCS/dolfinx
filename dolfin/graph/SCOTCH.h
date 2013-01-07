// Copyright (C) 2010 Garth N. Wells
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
// Last changed:

#ifndef __SCOTCH_PARTITIONER_H
#define __SCOTCH_PARTITIONER_H

#include <cstddef>
#include <set>
#include <vector>

#include "Graph.h"

namespace dolfin
{
  // Forward declarations
  class LocalMeshData;

  /// This class proivdes an interface to SCOTCH-PT (parallel version)

  class SCOTCH
  {
  public:

    // Compute cell partition
    static void compute_partition(std::vector<std::size_t>& cell_partition,
                                  const LocalMeshData& mesh_data);

    // Compute graph re-ordering
    static std::vector<std::size_t> compute_reordering(const Graph& graph);

    // Compute graph re-ordering
    static void compute_reordering(const Graph& graph,
                                   std::vector<std::size_t>& permutation,
                                   std::vector<std::size_t>& inverse_permutation);

  private:

    // Compute cell partitions from distribted dual graph
    static void partition(const std::vector<std::set<std::size_t> >& local_graph,
                          const std::set<std::size_t>& ghost_vertices,
                          const std::vector<std::size_t>& global_cell_indices,
                          const std::size_t num_global_vertices,
                          std::vector<std::size_t>& cell_partition);

  };

}

#endif
