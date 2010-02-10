// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#include <dolfin/common/Timer.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "SCOTCH.h"

#if defined HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif

using namespace dolfin;

#if defined HAS_SCOTCH
//-----------------------------------------------------------------------------
void SCOTCH::compute_partition(std::vector<uint>& cell_partition,
                               const LocalMeshData& mesh_data)
{
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
#else
void SCOTCH::compute_partition(std::vector<uint>& cell_partition, 
                       const LocalMeshData& data)
{
  error("SCOTCH::compute_partition requires SCOTCH.");
}
//-----------------------------------------------------------------------------
#endif
void compute_dual_graph(const LocalMeshData& mesh_data)
{
  // Compute distributed dual graph here
}
//-----------------------------------------------------------------------------

