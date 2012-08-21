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

#include <set>
#include <vector>

#include <dolfin/common/types.h>
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
    static void compute_partition(std::vector<uint>& cell_partition,
                                  const LocalMeshData& mesh_data);

    // Compute graph re-numbering
    static void compute_renumbering(const Graph& graph,
                                    std::vector<uint>& permutation,
                                    std::vector<uint>& inverse_permutation);

  private:

    // Compute cell partitions from distribted dual graph
    static void partition(const std::vector<std::set<uint> >& local_graph,
                          const std::set<uint>& ghost_vertices,
                          const std::vector<uint>& global_cell_indices,
                          const uint num_global_vertices,
                          std::vector<uint>& cell_partition);

  };

}

#endif
