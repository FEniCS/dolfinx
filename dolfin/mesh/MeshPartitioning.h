// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Niclas Jansson, 2008.
//
// First added:  2007-04-24
// Last changed: 2008-12-02

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

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

    // Partition vertices (geometric partitioning)
    static void partition_vertices(const LocalMeshData& data,
                                   std::vector<uint>& vertex_partition);

    // Distribute vertices according to "part" array
    static void distribute_vertices(LocalMeshData& data,
                                    const std::vector<uint>& vertex_partition);

    // Partition cells (topological partitioning)
    static void partition_cells();

  };

}

#endif
