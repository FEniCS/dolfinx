// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010
//
// First added:  2010-02-10
// Last changed:

#ifndef __PARMETIS_PARTITIONER_H
#define __PARMETIS_PARTITIONER_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{
  // Forward declarations
  class LocalMeshData;

  /// This class proivdes an interface to ParMETIS

  class ParMETIS
  {
  public:

    // Compute cell partition
    static void compute_partition(std::vector<uint>& cell_partition,
                                  const LocalMeshData& mesh_data);

  };

}

#endif
