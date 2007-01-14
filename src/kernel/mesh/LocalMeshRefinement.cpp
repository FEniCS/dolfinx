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
//  const uint num_edges = mesh.size(1);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Init new vertices and cells
  uint num_new_vertices = 0;
  uint num_new_cells = 0;

  // Initialise forbidden edges 
  MeshFunction<bool> edge_forbidden(mesh);  
  edge_forbidden.init(1);
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);

  // Initialise data for finding longest edge   
  Edge* longest_edge = 0;  
  real lmax, l;

  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c){
    if (cell_marker.get(*c) == true)
    {
      // Find longest edge of cell c
      lmax = 0.0;
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
	      if ( edge_forbidden.get(*e) == false )
        {
	        l = e->length();
	        if ( lmax < l )
          {
	          lmax = l;
	          longest_edge = &(*e); 
	        }
	      }
      }

      // If at least one edge should be bisected
      if ( lmax > 0.0 )
      {
        num_new_vertices++;
        num_new_cells++;
	      for (CellIterator cn(*longest_edge); !cn.end(); ++cn)
        {
	        // set markers of all cell neighbors of longest edge to false 
	        cell_marker.set(cn->index(),false);
	        // set all neighbors edges to forbidden
	        for (EdgeIterator en(*cn); !en.end(); ++en)
	          edge_forbidden.set(en->index(),true);
	      }
      }

    }
  }


  // Specify number of vertices and cells
  editor.initVertices(num_vertices + num_new_vertices);
  editor.initCells(num_cells + num_new_cells);

  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.addVertex(vertex++, v->point());

  /*

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

