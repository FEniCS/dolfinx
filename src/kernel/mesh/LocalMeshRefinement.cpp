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
void LocalMeshRefinement::refineSimplexMeshByBisection(Mesh& mesh, 
						       MeshFunction<bool>& cell_marker)
{
  dolfin_info("Refining simplicial mesh by bisection of edges.");

  // Check cell marker 
  // ...

  /*
  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);
  
  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);

  // Get cell type
  const CellType& cell_type = mesh.type();
  
  // Create new mesh and open for editing
  Mesh refined_mesh;
  MeshEditor editor;
  editor.open(refined_mesh, cell_type.cellType(),
	      mesh.topology().dim(), mesh.geometry().dim());
  
  // Get size of old mesh
  const uint num_vertices = mesh.size(0);
  const uint num_edges = mesh.size(1);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c){
    if (cell_marker.get(c) == true){
      






  for ( uint i < cell_marker.size() ){
      

    }
  }










  // Specify number of vertices and cells
  editor.initVertices(num_vertices + num_edges);
  editor.initCells(ipow(2, mesh.topology().dim())*num_cells);

  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.addVertex(vertex++, v->point());

  // Add new vertices
  for (EdgeIterator e(mesh); !e.end(); ++e)
    editor.addVertex(vertex++, e->midpoint());

  // Add cells
  uint current_cell = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_type.refineCell(*c, editor, current_cell);

  // Overwrite old mesh with refined mesh
  editor.close();
  mesh = refined_mesh;

  cout << "Refined mesh: " << mesh << endl;



  */

  dolfin_warning("Not implemented yet.");

  // Local mesh refinement of edge with nodes n1,n2 by node insertion: 
  // (1) Insert new node n_new on midpoint of edge. 
  // For all cells containing edge
  // (2) Delete old cell (2d: n1,n2,n3, 3d:n1,n2,n3,n4)
  // (3) Add new cells: (2d: n_new,n1,n3; n_new,n2,n3, 3d: n_new,n1,n3,n4; n_new,n2,n3,n4)
  // (4) Reset connectivity  
}
//-----------------------------------------------------------------------------

