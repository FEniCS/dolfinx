// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2007-01-16

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
#include "BoundaryMesh.h"
#include "LocalMeshRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineMeshByEdgeBisection(Mesh& mesh, 
                                                    MeshFunction<bool>& cell_marker,
                                                    bool refine_boundary)
{
  message("Refining simplicial mesh by edge bisection.");
  
  // Get size of old mesh
  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());
  
  // Check cell marker 
  if ( cell_marker.size() != num_cells ) error("Wrong dimension of cell_marker");
  
  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);
  
  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);
  
  // Get cell type
  const CellType& cell_type = mesh.type();
  
  // Init new vertices and cells
  uint num_new_vertices = 0;
  uint num_new_cells = 0;
  
  // Compute number of vertices and cells 
  /*
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if((cell_marker.get(*c) == true))
    {
      //cout << "marked cell: " << endl;
      //cout << c->midpoint() << endl;
      //cout << c->index() << endl;
    }
  }
  */

  // Create new mesh and open for editing
  Mesh refined_mesh;
  MeshEditor editor;
  editor.open(refined_mesh, cell_type.cellType(),
              mesh.topology().dim(), mesh.geometry().dim());
  
  // Initialize mappings
  Array<int> old2new_cell(mesh.numCells());
  Array<int> old2new_vertex(mesh.numVertices());

  // Initialise forbidden edges 
  MeshFunction<bool> edge_forbidden(mesh);  
  edge_forbidden.init(1);
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);
  
  // If refinement of boundary is forbidden 
  if ( !refine_boundary )
  {
    BoundaryMesh boundary(mesh);
    for (EdgeIterator e(boundary); !e.end(); ++e)
      edge_forbidden.set(e->index(),true);
  }

  // Initialise forbidden cells 
  MeshFunction<bool> cell_forbidden(mesh);  
  cell_forbidden.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(), false);
  
  // Initialise data for finding longest edge   
  uint longest_edge_index = 0;
  double lmax, l;

  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( cell_marker.get(*c) && !cell_forbidden.get(*c) )
    {
//       cout << "marked cell: " << endl;
//       cout << c->midpoint() << endl;
//       cout << c->index() << endl;

      // Find longest edge of cell c
      lmax = 0.0;
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        if ( !edge_forbidden.get(*e) )
        {
          l = e->length();
          if ( lmax < l )
          {
            lmax = l;
            longest_edge_index = e->index(); 
          }
        }
      }

      Edge longest_edge(mesh,longest_edge_index);

      // If at least one edge should be bisected
      if ( lmax > 0.0 )
      {
        // Add new vertex 
        num_new_vertices++;

        for (CellIterator cn(longest_edge); !cn.end(); ++cn)
        {
          if ( !cell_forbidden.get(*cn) )
          {
            // Count new cells
            num_new_cells++;
            // set markers of all cell neighbors of longest edge to false 
            //if ( cn->index() != c->index() ) 
            cell_forbidden.set(cn->index(),true);
            // set all the edges of cell neighbors to forbidden
            for (EdgeIterator en(*cn); !en.end(); ++en)
              edge_forbidden.set(en->index(),true);
          }
        }
      }
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices + num_new_vertices);
  editor.initCells(num_cells + num_new_cells);

  //cout << "Number of cells in old mesh: " << num_cells << "; to add: " << num_new_cells << endl;
  //cout << "Number of vertices in old mesh: " << num_vertices << "; to add: " << num_new_vertices << endl;
  
  // Add old vertices
  uint current_vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.addVertex(current_vertex++, v->point());

  // Add old unrefined cells 
  uint current_cell = 0;
  Array<uint> cell_vertices(cell_type.numEntities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    //if ( (cell_marker.get(*c) == false) && (cell_forbidden.get(*c) == false) )
    if ( !cell_forbidden.get(*c) )
    {
      uint cv = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
        cell_vertices[cv++] = v->index(); 
      editor.addCell(current_cell++, cell_vertices);
    }
  }
  
  // Reset forbidden edges 
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(), false);

  // If refinement of boundary is forbidden 
  if ( !refine_boundary )
  {
    BoundaryMesh boundary(mesh);
    for (EdgeIterator e(boundary); !e.end(); ++e)
      edge_forbidden.set(e->index(),true);
  }

  // Reset forbidden cells 
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);

  // Add new vertices and cells. 
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( cell_marker.get(*c) && !cell_forbidden.get(*c) )
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
            longest_edge_index = e->index(); 
          }
        }
      }
     
      Edge longest_edge(mesh,longest_edge_index);

      // If at least one edge should be bisected
      if ( lmax > 0.0 )
      {
        // Add new vertex
        editor.addVertex(current_vertex++, longest_edge.midpoint());

       for (CellIterator cn(longest_edge); !cn.end(); ++cn)
       {
         // Add new cell
         bisectEdgeOfSimplexCell(*cn, longest_edge, current_vertex, editor, current_cell);

         // set markers of all cell neighbors of longest edge to false 
         //if ( cn->index() != c->index() ) 
         cell_marker.set(cn->index(),false);
         // set all edges of cell neighbors to forbidden
         for (EdgeIterator en(*cn); !en.end(); ++en)
           edge_forbidden.set(en->index(), true);
        }
      }
    }
  }

  // Overwrite old mesh with refined mesh
  editor.close();
  mesh = refined_mesh;
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::bisectEdgeOfSimplexCell(const Cell& cell, 
                                                  Edge& edge, 
                                                  uint& new_vertex, 
                                                  MeshEditor& editor, 
                                                  uint& current_cell) 
{
  // Init cell vertices 
  Array<uint> cell1_vertices(cell.numEntities(0));
  Array<uint> cell2_vertices(cell.numEntities(0));

  // Get edge vertices 
  const uint* edge_vert = edge.entities(0);

  uint vc1 = 0;
  uint vc2 = 0;

  for (VertexIterator v(cell); !v.end(); ++v)
  {
    if ( (v->index() != edge_vert[0]) && (v->index() != edge_vert[1]) )
    {
      cell1_vertices[vc1++] = v->index();
      cell2_vertices[vc2++] = v->index();
    }
  }

  cell1_vertices[vc1++] = new_vertex - 1; 
  cell2_vertices[vc2++] = new_vertex - 1; 
  
  cell1_vertices[vc1++] = edge_vert[0]; 
  cell2_vertices[vc2++] = edge_vert[1]; 

  editor.addCell(current_cell++, cell1_vertices);
  editor.addCell(current_cell++, cell2_vertices);

}
//-----------------------------------------------------------------------------

