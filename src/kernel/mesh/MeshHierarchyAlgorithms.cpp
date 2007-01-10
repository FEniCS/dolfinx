// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-09

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
#include <dolfin/MeshHierarchy.h>
#include <dolfin/MeshHierarchyAlgorithms.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::refineTetMesh(MeshHierarchy& mesh_hierarchy) 
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
void MeshHierarchyAlgorithms::evaluateMarks(Mesh& mesh) 
{
  // This function implements the "EvaluateMarks" subroutine by Bey 
  dolfin_warning("Not implemented yet.");

  /*
    cell_children = new MeshFunction<uint>[num_meshes]; 
    cell_children[0].init(2);

    delete [] cell_children;



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
void MeshHierarchyAlgorithms::closeMesh(Mesh& mesh) 
{
  // This function implements the "CloseGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::closeCell(Cell& cell) 
{
  // This function implements the "CloseElement" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::unrefineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "UnrefineGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::refineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "RefineGrid" subroutine by Bey 
  dolfin_warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------

