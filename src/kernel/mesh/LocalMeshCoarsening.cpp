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
#include <dolfin/BoundaryMesh.h>
#include <dolfin/LocalMeshCoarsening.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshCoarsening::coarsenMeshByEdgeCollapse(Mesh& mesh, 
                                                    MeshFunction<bool>& cell_marker,
                                                    bool coarsen_boundary)
{
  dolfin_info("Coarsen simplicial mesh by edge collapse.");

  // Get size of old mesh
  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());
  
  // Check cell marker 
  if ( cell_marker.size() != num_cells ) dolfin_error("Wrong dimension of cell_marker");
  
  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);
  
  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);
  
  // Get cell type
  const CellType& cell_type = mesh.type();
  
  // Create new mesh and open for editing
  Mesh coarse_mesh;
  MeshEditor editor;
  editor.open(coarse_mesh, cell_type.cellType(),
	      mesh.topology().dim(), mesh.geometry().dim());
  
  // Init new vertices and cells
  uint num_vertices_to_remove = 0;
  uint num_cells_to_remove = 0;
  
  // Initialise vertices to remove 
  MeshFunction<bool> vertex_to_remove_index(mesh);  
  vertex_to_remove_index.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_to_remove_index.set(v->index(),false);

  MeshFunction<bool> cell_to_remove(mesh);  
  cell_to_remove.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_to_remove.set(c->index(),false);

  // Initialise forbidden verticies   
  MeshFunction<bool> vertex_forbidden(mesh);  
  vertex_forbidden.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden.set(v->index(),false);

  // If coarsen boundary is forbidden
  if ( coarsen_boundary == false )
  {
    BoundaryMesh boundary(mesh);
    for (VertexIterator v(boundary); !v.end(); ++v)
      vertex_forbidden.set(v->index(),true);
  }

  // Initialise forbidden edges 
  MeshFunction<bool> edge_forbidden(mesh);  
  edge_forbidden.init(1);
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);

  // Initialise forbidden cells 
  MeshFunction<bool> cell_forbidden(mesh);  
  cell_forbidden.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);
  
  // Initialise data for finding which vertex to remove   
  uint* edge_vertex;
  uint shortest_edge_index = 0;
  real lmin, l;

  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c)
  {

    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {

      // Find shortest edge of cell c
      lmin = 1.0e10 * c->diameter();
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        if ( edge_forbidden.get(*e) == true ) break;

        l = e->length();
        if ( lmin > l )
        {
          lmin = l;
          shortest_edge_index = e->index(); 
        }
      }

      // If at least one vertex should be removed 
      if ( lmin < 0.5e10 * c->diameter() )
      {
        Edge shortest_edge(mesh,shortest_edge_index);

        uint vert2remove_idx = 0;
        edge_vertex = shortest_edge.entities(0);
        if ( (vertex_forbidden.get(edge_vertex[0]) == true) && 
             (vertex_forbidden.get(edge_vertex[1]) == true) ) 
          break;
        if ( vertex_forbidden.get(edge_vertex[0]) == true )
        {
          vert2remove_idx = edge_vertex[1];
          vertex_to_remove_index.set(edge_vertex[1],true);
        }
        else if ( vertex_forbidden.get(edge_vertex[1]) == true )
        {
          vert2remove_idx = edge_vertex[0];
          vertex_to_remove_index.set(edge_vertex[0],true);
        }
        else if ( edge_vertex[0] > edge_vertex[1] ) 
        {
          vert2remove_idx = edge_vertex[0];
          vertex_to_remove_index.set(edge_vertex[0],true);
        }
        else
        {
          vert2remove_idx = edge_vertex[1];
          vertex_to_remove_index.set(edge_vertex[1],true);
        }       
        Vertex vertex_to_remove(mesh,vert2remove_idx);
        
	// Remove vertex 
	num_vertices_to_remove++;

        // Set cells of vertex to remove to forbidden 
        for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
          cell_forbidden.set(cn->index(),true);

	for (CellIterator cn(shortest_edge); !cn.end(); ++cn)
	{
          // remove cell
          cell_to_remove.set(cn->index(),true);
	  num_cells_to_remove++;

          /*
          // Set cells of edge to remove to forbidden 
          cell_forbidden.set(cn->index(),true);
          */

	  // set all the edges of the neighbor cells to forbidden
	  for (EdgeIterator e(*cn); !e.end(); ++e)
	    edge_forbidden.set(e->index(),true);
	}
      }
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices - num_vertices_to_remove);
  editor.initCells(num_cells - num_cells_to_remove);

  cout << "no old cells: " << num_cells << ", cells to remove: " << num_cells_to_remove << endl;
  cout << "no old vert: " << num_vertices << ", vertices to remove: " << num_vertices_to_remove << endl;
  
  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    if ( vertex_to_remove_index.get(*v) == false ) 
      editor.addVertex(vertex++, v->point());
  }

  // Add old unrefined cells 
  uint cv_idx;
  uint current_cell = 0;
  Array<uint> cell_vertices(cell_type.numEntities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    //if ( (cell_marker.get(*c) == false) && (cell_forbidden.get(*c) == false) )
    if ( cell_forbidden.get(*c) == false )
    {
      cv_idx = 0;
      for (VertexIterator v(c); !v.end(); ++v)
        cell_vertices[cv_idx++] = v->index(); 
      editor.addCell(current_cell++, cell_vertices);
    }
  }
  
  // Reset forbidden edges 
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);

  // Reset forbidden cells 
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);


  // Add new vertices and cells. 
  for (CellIterator c(mesh); !c.end(); ++c)
  {

    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {

      // Find shortest edge of cell c
      lmin = 1.0e10 * c->diameter();
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        if ( edge_forbidden.get(*e) == true ) break;

        l = e->length();
        if ( lmin > l )
        {
          lmin = l;
          shortest_edge_index = e->index(); 
        }
      }

      // If at least one vertex should be removed 
      if ( lmin < 0.5e10 * c->diameter() )
      {
        Edge shortest_edge(mesh,shortest_edge_index);

        uint vert2remove_idx = 0;
        edge_vertex = shortest_edge.entities(0);
        if ( (vertex_forbidden.get(edge_vertex[0]) == true) && 
             (vertex_forbidden.get(edge_vertex[1]) == true) ) 
          break;
        if ( vertex_forbidden.get(edge_vertex[0]) == true )
        {
          vert2remove_idx = edge_vertex[1];
          vertex_to_remove_index.set(edge_vertex[1],true);
        }
        else if ( vertex_forbidden.get(edge_vertex[1]) == true )
        {
          vert2remove_idx = edge_vertex[0];
          vertex_to_remove_index.set(edge_vertex[0],true);
        }
        else if ( edge_vertex[0] > edge_vertex[1] ) 
        {
          vert2remove_idx = edge_vertex[0];
          vertex_to_remove_index.set(edge_vertex[0],true);
        }
        else
        {
          vert2remove_idx = edge_vertex[1];
          vertex_to_remove_index.set(edge_vertex[1],true);
        }       
        Vertex vertex_to_remove(mesh,vert2remove_idx);
        
	// Remove vertex 
        collapseEdge(mesh, shortest_edge, vertex_to_remove, cell_to_remove, editor, current_cell);

        // Set cells of vertex to remove to forbidden 
        for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
          cell_forbidden.set(cn->index(),true);

	for (CellIterator cn(shortest_edge); !cn.end(); ++cn)
        {
          /*
          // Set cells of edge to remove to forbidden 
          cell_forbidden.set(cn->index(),true);
          */

          // set all the edges of the neighbor cells to forbidden
          for (EdgeIterator e(*cn); !e.end(); ++e)
	    edge_forbidden.set(e->index(),true);
	}
      }
    }
  }

  // Overwrite old mesh with refined mesh
  editor.close();
  mesh = coarse_mesh;

}
//-----------------------------------------------------------------------------
void LocalMeshCoarsening::collapseEdge(Mesh& mesh, Edge& edge, 
                                       Vertex& vertex_to_remove, 
                                       MeshFunction<bool>& cell_to_remove, 
                                       MeshEditor& editor, 
                                       uint& current_cell) 
{

  const CellType& cell_type = mesh.type();
  Array<uint> cell_vertices(cell_type.numEntities(0));

  uint vert_slave = vertex_to_remove.index();
  uint vert_master = 0; 
  uint* edge_vertex = edge.entities(0);
  if ( edge_vertex[0] == vert_slave ) 
    vert_master = edge_vertex[1]; 
  else
    vert_master = edge_vertex[0]; 

  uint cv_idx;
  for (CellIterator c(vertex_to_remove); !c.end(); ++c)
  {
    if ( cell_to_remove.get(*c) == false ) 
    {
      cv_idx = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
      {  
        if ( v->index() == vert_slave ) cell_vertices[cv_idx++] = vert_master; 
        else                            cell_vertices[cv_idx++] = v->index();
      }
      editor.addCell(current_cell++, cell_vertices);
    }    
  }
  
}
//-----------------------------------------------------------------------------



