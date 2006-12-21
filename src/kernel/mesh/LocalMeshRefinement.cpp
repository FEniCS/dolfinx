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
#include <dolfin/MeshHierarchy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineTetMesh(MeshHierarchy& mesh_hierarchy) 
{
  // This function implements the "GlobalRefinement" algorithm by Bey 

  uint num_meshes = uint(mesh_hierarchy.size());

  for (uint k = num_meshes-1; k >= 0; k--)
  {
    /*
    evaluateMarks(mesh_hierarchy(k));
    closeMesh(mesh_hierarchy(k));
    */
  }

  for (uint k = 0; k < num_meshes; k++)
  {
    if ( mesh_hierarchy(k).numCells() > 0 )
    {
      /*
      if ( k > 0 ) closeMesh(mesh_hierarchy(k));
      unrefineMesh(mesh_hierarchy,k);
      refineMesh(mesh_hierarchy,k);
      */
    }
  }

  /*
  if      ( mesh_hierarchy(num_meshes-1).numCells() == 0 ) num_meshes--;
  else if ( mesh_hierarchy(num_meshes).numCells()   != 0 ) num_meshes++;
  */
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::evaluateMarks(Mesh& mesh) 
{
  // This function implements the "EvaluateMarks" subroutine by Bey 
  dolfin_warning("Not implemented yet.");

  /*
  MeshFunction<uint> cell_marker(mesh); 
  MeshFunction<uint> cell_state(mesh); 
  cell_marker.init(2);
  cell_state.init(2);

  MeshFunction<uint> edge_marker(mesh); 
  MeshFunction<uint> edge_state(mesh); 
  edge_marker.init(1);
  edge_state.init(1);
  */
  
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::closeMesh(Mesh& mesh) 
{
  // This function implements the "CloseGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::closeCell(Cell& cell) 
{
  // This function implements the "CloseElement" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::unrefineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "UnrefineGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "RefineGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
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

