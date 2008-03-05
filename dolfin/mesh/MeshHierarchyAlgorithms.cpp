// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-09

#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshEditor.h"
#include "MeshFunction.h"
#include "Vertex.h"
#include "Edge.h"
#include "Cell.h"
#include "MeshHierarchy.h"
#include "MeshHierarchyAlgorithms.h"

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
  warning("Not implemented yet.");

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
  warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::closeCell(Cell& cell) 
{
  // This function implements the "CloseElement" subroutine by Bey 
  warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::unrefineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "UnrefineGrid" subroutine by Bey 
  warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------
void MeshHierarchyAlgorithms::refineMesh(MeshHierarchy& mesh, uint k)
{
  // This function implements the "RefineGrid" subroutine by Bey 
  warning("Not implemented yet.");
}
//-----------------------------------------------------------------------------

