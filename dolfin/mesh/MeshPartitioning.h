// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2008-12-15

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class LocalMeshData;

  /// This class partitions and distributes a mesh based on
  /// partitioned local mesh data. Note that the local mesh data will
  /// also be repartitioned and redistributed during the computation
  /// of the mesh partitioning.

  class MeshPartitioning
  {
  public:

    /// Create a partitioned mesh based on partitioned local mesh data
    static void partition(Mesh& mesh, LocalMeshData& data);

  private:

    // Compute cell partition
    static void compute_partition(std::vector<uint>& cell_partition,
                                  const LocalMeshData& data);

    // Distribute cells
    static void distribute_cells(LocalMeshData& data,
                                 const std::vector<uint>& cell_partition);

    // Distribute vertices 
    static void distribute_vertices(LocalMeshData& data,
                                    std::map<uint, uint>& glob2loc);

    // Build mesh
    static void build_mesh(Mesh& mesh, const LocalMeshData& data,
                           std::map<uint, uint>& glob2loc);

  };

}

#endif
