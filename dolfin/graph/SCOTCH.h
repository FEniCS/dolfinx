// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#ifndef __SCOTCH_PARTITIONER_H
#define __SCOTCH_PARTITIONER_H

#include <set>
#include <vector>

#include <dolfin/common/types.h>

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
