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
#include <dolfin/LocalMeshCoarsening.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshCoarsening::coarsenSimplexMeshByEdgeCollapse(Mesh& mesh, 
                                                           MeshFunction<bool>& cell_marker)
{
  dolfin_info("Coarsen simplicial mesh by node deletion/edge collapse.");



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
  
  // Get size of old mesh
  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());
  
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

  // Initialise forbidden vertices 
  MeshFunction<bool> vertex_forbidden(mesh);  
  vertex_forbidden.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden.set(v->index(),false);

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
  Vertex* vertex_to_remove = 0;  
  Edge* shortest_edge = 0;  
  uint* edge_vertices;
  uint shortest_edge_index = 0;
  real lmin, l;

  for (VertexIterator v(mesh); !v.end(); ++v)
    cout << "Vertex " << v->index() << ": " << vertex_forbidden.get(*v) << endl;
  
  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c)
  {

    cout << "cell " << c->index() << ": " << cell_marker.get(*c) << ", nodes = ";
    for (VertexIterator v(c); !v.end(); ++v)
      cout << v->index() << " ";
    cout << endl;

    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      cout << "Edge " << e->index() << ": " << edge_forbidden.get(*e) << " = ";
      for (VertexIterator v(e); !v.end(); ++v)
	cout << v->index() << " ";
      cout << endl;
    }
    
    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {
      // Find shortest edge of cell c
      lmin = 1.0e10 * c->diameter();

      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
          if ( vertex_forbidden.get(*v) == true ) break;
        
        if ( edge_forbidden.get(*e) == true ) break;

        l = e->length();
        if ( lmin > l )
        {
          lmin = l;
          shortest_edge_index = e->index(); 
        }
        cout << "l = " << l << ", lmin = " << lmin << endl;
      }
     
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
          if ( vertex_forbidden.get(*v) == true ) break;
        
        if ( edge_forbidden.get(*e) == true ) break;
        
        edge_vertices = e->entities(0);
        dolfin_assert(edge_vertices);
          
        for (VertexIterator v(*e); !v.end(); ++v)
        { 
          vertex_to_remove = &(*v); 
          if ( edge_vertices[0] > edge_vertices[1] )
            break;
        }
        shortest_edge = &(*e);
        break;
      }
      
      vertex_to_remove_index.set(vertex_to_remove->index(),true);
     
      cout << "shortest edge: " << shortest_edge->index() << " = ";
      for (VertexIterator v(*shortest_edge); !v.end(); ++v)
	cout << v->index() << " "; 
      cout << " : vertex to remove = " << vertex_to_remove->index() << endl;

      // If at least one vertex should be removed 
      if ( lmin < 0.5e10 * c->diameter() )
      {
	// Remove vertex 
	cout << "remove vertex" << endl;
	num_vertices_to_remove++;

        for (VertexIterator v(*vertex_to_remove); !v.end(); ++v)
        {
          vertex_forbidden.set(v->index(),true);
          cout << "Set to forbidden vertex: " << v->index() << endl;
        }
        for (CellIterator cn(*vertex_to_remove); !cn.end(); ++cn)
        {
          if ( cn->index() != c->index() )
          { 
            cell_forbidden.set(cn->index(),true);
            cout << "Set to forbidden cell: " << cn->index() << endl;
          }
        }
	for (CellIterator cn(*shortest_edge); !cn.end(); ++cn)
	{
	  // Cells to remove 
          cell_to_remove.set(cn->index(),true);
	  cout << "cells to remove" << endl;
	  num_cells_to_remove++;

	  // set all the edges of the neighbor cells to forbidden
	  for (EdgeIterator e(*cn); !e.end(); ++e)
          {
	    edge_forbidden.set(e->index(),true);
	    cout << "Set to forbidden edge: " << e->index() << endl;
          }


	}
      }
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices - num_vertices_to_remove);
  editor.initCells(num_cells - num_cells_to_remove);

  cout << "no old cells: " << num_cells << ", cells to remove: " << num_cells_to_remove << endl;
  cout << "no old vert: " << num_vertices << ", vertices to remove: " << num_vertices_to_remove << endl;
  
  coarse_mesh.disp();

  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    if ( vertex_to_remove_index.get(*v) == false ) 
      editor.addVertex(vertex++, v->point());
  }

  coarse_mesh.disp();

  // Add old unrefined cells 
  uint cv;
  uint current_cell = 0;
  Array<uint> cell_vertices(cell_type.numEntities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( (cell_marker.get(*c) == false) && (cell_forbidden.get(*c) == false) )
    {
      cv = 0;
      for (VertexIterator v(c); !v.end(); ++v)
        cell_vertices[cv++] = v->index(); 
      editor.addCell(current_cell++, cell_vertices);
    }
  }
  
  coarse_mesh.disp();

  // Reset forbidden vertices 
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden.set(v->index(),false);

  // Reset forbidden edges 
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);

  // Reset forbidden cells 
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);


  // Add new vertices and cells. 
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    cout << "Cell " << c->index() << ": current_cell = " << current_cell << endl;
    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {
      // Find shortest edge of cell c
      lmin = 1.0e10 * c->diameter();

      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
          if ( vertex_forbidden.get(*v) == true ) break;
        
        if ( edge_forbidden.get(*e) == true ) break;

        l = e->length();
        if ( lmin > l )
        {
          lmin = l;
          shortest_edge_index = e->index(); 
        }
        cout << "l = " << l << ", lmin = " << lmin << endl;
      }
     
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
          if ( vertex_forbidden.get(*v) == true ) break;
        
        if ( edge_forbidden.get(*e) == true ) break;

        edge_vertices = e->entities(0);
        dolfin_assert(edge_vertices);
          
        for (VertexIterator v(*e); !v.end(); ++v)
        { 
          vertex_to_remove = &(*v); 
          if ( edge_vertices[0] > edge_vertices[1] )
            break;
        }
        shortest_edge = &(*e);
        break;
      }
      
      vertex_to_remove_index.set(vertex_to_remove->index(),true);
     
      cout << "shortest edge: " << shortest_edge->index() << " = ";
      for (VertexIterator v(*shortest_edge); !v.end(); ++v)
	cout << v->index() << " "; 
      cout << " : vertex to remove = " << vertex_to_remove->index() << endl;

      // If at least one vertex should be removed 
      if ( lmin < 0.5e10 * c->diameter() )
      {
        // Remove vertex 
        collapseEdgeByNodeDeletion(mesh, *shortest_edge, *vertex_to_remove, cell_to_remove, editor, current_cell);

        for (VertexIterator v(*vertex_to_remove); !v.end(); ++v)
        {
          vertex_forbidden.set(v->index(),true);
          cout << "Set to forbidden vertex: " << v->index() << endl;
        }
        for (CellIterator cn(*vertex_to_remove); !cn.end(); ++cn)
        {
          if ( cn->index() != c->index() )
          { 
            cell_forbidden.set(cn->index(),true);
            cout << "Set to forbidden cell: " << cn->index() << endl;
          }
        }
	for (CellIterator cn(*shortest_edge); !cn.end(); ++cn)
	{
	  // Cells to remove 
          cell_to_remove.set(cn->index(),true);
	  cout << "cells to remove" << endl;
	  num_cells_to_remove++;

	  // set all the edges of the neighbor cells to forbidden
	  for (EdgeIterator e(*cn); !e.end(); ++e)
          {
	    edge_forbidden.set(e->index(),true);
	    cout << "Set to forbidden edge: " << e->index() << endl;
          }
	}
      }
    }
    cout << "Cell " << c->index() << ": current_cell = " << current_cell << endl;
  }

  // Overwrite old mesh with refined mesh
  editor.close();
  coarse_mesh.disp();

  mesh = coarse_mesh;
  
  cout << "Coarsened mesh: " << mesh << endl;  



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
void LocalMeshCoarsening::collapseEdgeByNodeDeletion(Mesh& mesh, Edge& edge, 
                                                     Vertex& vertex_to_remove, 
                                                     MeshFunction<bool>& cell_to_remove, 
                                                     MeshEditor& editor, 
                                                     uint& current_cell) 
{

  cout << "Coarsen mesh by edge collapse" << endl; 

  const CellType& cell_type = mesh.type();
  Array<uint> cell_vertices(cell_type.numEntities(0));

  const uint* cv;
  uint cv_idx;

  uint vert_slave = vertex_to_remove.index();
  uint vert_master = 0; 
  for (VertexIterator v(edge); !v.end(); ++v)
  {
    if ( v->index() != vertex_to_remove.index() ) vert_master = v->index();
  }

  for (CellIterator c(vertex_to_remove); !c.end(); ++c)
  {
    if ( cell_to_remove.get(*c) == false ) 
    {
      cv_idx = 0;
      cv = c->entities(0);
      for (VertexIterator v(*c); !v.end(); ++v)
      {  
        if ( v->index() == vert_slave ) cell_vertices[cv_idx++] = vert_master; 
        else                            cell_vertices[cv_idx++] = v->index();
      }
      cout << "cell vert ="; 
      for (int i=0; i<3; i++) cout << " " << cell_vertices[i];
      cout << endl;
      editor.addCell(current_cell++, cell_vertices);
      cout << "cell added" << endl;
    }    
  }

}
//-----------------------------------------------------------------------------



