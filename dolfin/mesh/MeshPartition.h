// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-04-24
// Last changed: 2008-02-04

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <dolfin/common/types.h>

namespace dolfin
{
  /// This class provides a set of functions to partition a Mesh

  class MeshPartition
  {
  public:

    /// Partition a mesh into num_part partitions
    static void partition(Mesh& mesh, MeshFunction<uint>& partitions, uint num_partitions);

  };

}

#endif
