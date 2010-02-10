// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#ifndef __SCOTCH_PARTITIONER_H
#define __SCOTCH_PARTITIONER_H

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

    // Compute distribted dual graph for mesh
    static void compute_dual_graph(const LocalMeshData& mesh_data);

  };

}

#endif
