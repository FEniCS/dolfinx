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
  const uint num_cells = mesh.size(mesh.topology().dim());
  
  // Init new vertices and cells
  uint num_new_vertices = 0;
  uint num_new_cells = 0;
  
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
  
  // Initialise data for finding longest edge   
  Edge* longest_edge = 0;  
  uint longest_edge_index = 0;
  real lmax, l;

  for (EdgeIterator e(mesh); !e.end(); ++e)
  {
    cout << "Edge " << e->index() << ": " << edge_forbidden.get(*e) << " = ";
    for (VertexIterator v(e); !v.end(); ++v)
      cout << v->index() << " ";
    cout << endl;
  }
  
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
	  cout << "l = " << l << ", lmax = " << lmax << endl;
	}
      }
     
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
	if ( e->index() == longest_edge_index )
	{
	  longest_edge = &(*e); 
	  break;
	}
      }
     
      cout << "longest edge: " << longest_edge->index() << " = ";
      for (VertexIterator v(*longest_edge); !v.end(); ++v)
	cout << v->index() << " "; 
      cout << endl;

      // If at least one edge should be bisected
      if ( lmax > 0.0 )
      {
	// Add new vertex 
	cout << "add new node" << endl;
	num_new_vertices++;
	for (CellIterator cn(*longest_edge); !cn.end(); ++cn)
	{
	  // Add new cell
	  cout << "add new cell" << endl;
	  num_new_cells++;
	  // set markers of all cell neighbors of longest edge to false 
	  if ( cn->index() != c->index() ) cell_forbidden.set(cn->index(),true);
	  // set all the edges of cell neighbors to forbidden
	  for (EdgeIterator en(*cn); !en.end(); ++en)
          {
	    edge_forbidden.set(en->index(),true);
	    cout << "Set to forbidden edge: " << en->index() << " = ";
	    for (VertexIterator v(*en); !v.end(); ++v)
	      cout << v->index() << " "; 
	    cout << endl;
          }
	}
      }
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices + num_new_vertices);
  editor.initCells(num_cells + num_new_cells);

  cout << "no old cells: " << num_cells << ", new cells: " << num_new_cells << endl;
  cout << "no old vert: " << num_vertices << ", new vert: " << num_new_vertices << endl;
  
  refined_mesh.disp();

  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.addVertex(vertex++, v->point());

  refined_mesh.disp();

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
  
  // Reset forbidden edges 
  for (EdgeIterator e(mesh); !e.end(); ++e)
    edge_forbidden.set(e->index(),false);

  refined_mesh.disp();

  // Add new vertices and cells. 
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    cout << "Cell " << c->index() << ": current_cell = " << current_cell << endl;
    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
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
	  cout << "l = " << l << ", lmax = " << lmax << endl;
	}
      }
     
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
	if ( e->index() == longest_edge_index )
	{
	  longest_edge = &(*e); 
	  break;
	}
      }
     
      // If at least one edge should be bisected
      if ( lmax > 0.0 )
      {
	// Add new vertex
	editor.addVertex(vertex++, longest_edge->midpoint());

	for (CellIterator cn(*longest_edge); !cn.end(); ++cn)
	{
	  // Add new cell
	  bisectSimplexCell(*cn, *longest_edge, vertex, editor, current_cell);

	  // set markers of all cell neighbors of longest edge to false 
	  //if ( cn->index() != c->index() ) cell_marker.set(cn->index(),false);
	  // set all edges of cell neighbors to forbidden
	  for (EdgeIterator en(*cn); !en.end(); ++en)
	    edge_forbidden.set(en->index(),true);
	}
      }
    }
    cout << "Cell " << c->index() << ": current_cell = " << current_cell << endl;
  }

  // Overwrite old mesh with refined mesh
  editor.close();
  refined_mesh.disp();

  mesh = refined_mesh;
  
  cout << "Refined mesh: " << mesh << endl;  
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::bisectSimplexCell(Cell& cell, Edge& edge, uint& new_vertex, 
					    MeshEditor& editor, 
					    uint& current_cell) 
{
  Array<uint> cell1_vertices(cell.numEntities(0));
  Array<uint> cell2_vertices(cell.numEntities(0));

  cout << "size of cell = " << cell.numEntities(0) << endl;

  // Get edge vertices 
  const uint* v = edge.entities(0);
  dolfin_assert(v);

  // Compute indices for the edge vertices
  const uint v0 = v[0];
  const uint v1 = v[1];

  cout << "bisect edge: " << v0 << " " << v1 << endl;

  uint vc1 = 0;
  uint vc2 = 0;

  for (VertexIterator v(cell); !v.end(); ++v)
  {
    cout << "cell node: " << v->index() << endl;
    if ( (v->index() != v0) && (v->index() != v1) )
    {
      cout << "non bisected cell node: " << v->index() << endl;
      cell1_vertices[vc1++] = v->index();
      cell2_vertices[vc2++] = v->index();
    }
  }

  cell1_vertices[vc1++] = new_vertex - 1; 
  cell2_vertices[vc2++] = new_vertex - 1; 
  
  cell1_vertices[vc1++] = v0; 
  cell2_vertices[vc2++] = v1; 

  cout << "indices: " << vc1 << " " << vc2 << endl; 

  cout << "Bisected cell 1: " << cell1_vertices[0] << " "
       << cell1_vertices[1] << " "  << cell1_vertices[2] << " "  
       << cell1_vertices[3] << endl;

  cout << "Bisected cell 2: " << cell2_vertices[0] << " "
       << cell2_vertices[1] << " "  << cell2_vertices[2] << " "  
       << cell2_vertices[3] << endl;


  cout << "check 1: current cell: " << current_cell << endl; 
  editor.addCell(current_cell++, cell1_vertices);
  cout << "check 2: current cell: " << current_cell << endl; 
  editor.addCell(current_cell++, cell2_vertices);
  cout << "check 3: current cell: " << current_cell << endl; 

}
//-----------------------------------------------------------------------------

