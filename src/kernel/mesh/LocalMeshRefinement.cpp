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
#include <dolfin/MeshFunction.h>
#include <dolfin/Vertex.h>
#include <dolfin/Edge.h>
#include <dolfin/Cell.h>
#include <dolfin/LocalMeshRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineTetMesh(Mesh& mesh)
{
  MeshFunction<uint> cell_marker(mesh); 
  MeshFunction<uint> cell_state(mesh); 
  cell_marker.init(2);
  cell_state.init(2);

  MeshFunction<uint> edge_marker(mesh); 
  MeshFunction<uint> edge_state(mesh); 
  edge_marker.init(1);
  edge_state.init(1);
  
  
  

  


}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineSimplexByNodeInsertion(Mesh& mesh)
{
  dolfin_info("Refining edge in simplicial mesh by node insertion.");

  dolfin_warning("Not implemented yet.");

  // Local mesh refinement of edge with nodes n1,n2 by node insertion: 
  // (1) Insert new node n_new on midpoint of edge. 
  // For all cells containing edge
  // (2) Delete old cell (2d: n1,n2,n3, 3d:n1,n2,n3,n4)
  // (3) Add new cells: (2d: n_new,n1,n3; n_new,n2,n3, 3d: n_new,n1,n3,n4; n_new,n2,n3,n4)
  // (4) Reset connectivity  
}
//-----------------------------------------------------------------------------

