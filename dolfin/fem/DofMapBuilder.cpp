// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-12
// Last changed: 2008-08-12

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/main/MPI.h>
#include "UFC.h"
#include "DofMap.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dof_map, UFC& ufc, Mesh& mesh)
{
  // Work in progress, to be based on Algorithm 5 in the paper
  // http://home.simula.no/~logg/pub/papers/submitted-Log2008a.pdf

  message("Building parallel dof map (in parallel)");

  // Check that dof map has not been built
  if (dof_map.dof_map)
    error("Local-to-global mapping has already been computed.");

  // Allocate dof map
  const uint n = dof_map.local_dimension();
  dof_map.dof_map = new uint[n*mesh.numCells()];
  
  error("Not implemented.");

  // Get mesh functions
  //MeshFunction<uint>* S = mesh.data().meshFunction("subdomain overlap");
  //MeshFunction<uint>* F = mesh.data().meshFunction("facet overlap");
  //dolfin_assert(S);
  //dolfin_assert(F);

  // Get number of this process
  const uint this_process = MPI::process_number();
  message("Building dof map on processor %d.", this_process);

  // Build stage 0: Compute offsets
  computeOffsets(this_process);
  
  // Build stage 0.5: Communicate offsets
  communicateOffsets();
  
  // Build stage 1: Compute mapping on shared facets
  computeShared();
  
  // Build stage 2: Communicate mapping on shared facets
  communicateShared();
  
  // Build stage 3: Compute mapping for interior degrees of freedom
  computeInterior();
}
//-----------------------------------------------------------------------------
void DofMapBuilder::computeOffsets(uint this_process)
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicateOffsets()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::computeShared()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicateShared()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::computeInterior()
{

}
//-----------------------------------------------------------------------------
