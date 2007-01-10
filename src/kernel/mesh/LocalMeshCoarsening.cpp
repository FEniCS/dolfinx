// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/Vertex.h>
#include <dolfin/Edge.h>
#include <dolfin/Cell.h>
#include <dolfin/LocalMeshCoarsening.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshCoarsening::coarsenSimplexByNodeDeletion(Mesh& mesh, Edge& edge)
{
  dolfin_info("Coarsen simplicial mesh by node deletion/edge collapse.");

  dolfin_warning("Not implemented yet.");

  // Local mesh coarsening by node deletion/collapse of edge with nodes n1,n2: 
  // Independent of 2d/3d
  // For all cells containing edge: 
  // (1) Delete cell 
  // (2) n2 -> n1 
  // (3) Delete n2 
  // (4) Reset connectivity  
  
}
//-----------------------------------------------------------------------------
