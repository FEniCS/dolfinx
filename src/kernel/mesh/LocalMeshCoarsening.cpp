// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01

#include <list>

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
#include <dolfin/MeshGeometry.h>
#include <dolfin/LocalMeshCoarsening.h>
#include <dolfin/CellType.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>

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
  
  // Create new mesh
  Mesh coarse_mesh(mesh);
  
  // Initialise forbidden cells 
  MeshFunction<bool> cell_forbidden(mesh);  
  cell_forbidden.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);
  
  // Init new vertices and cells
  Array<int> old2new_cell(mesh.numCells());
  Array<int> old2new_vertex(mesh.numVertices());

  std::list<int> cells_to_coarsen(0);

  // Compute cells to delete
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (cell_marker.get(*c) == true)
    {
      cells_to_coarsen.push_back(c->index());
      //cout << "coarsen midpoint: " << c->midpoint() << endl;
    }
  }

  // Define cell mapping between old and new mesh
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    old2new_cell[c->index()] = c->index();
  }

  bool improving = true;

  while(improving)
  {

    int presize = cells_to_coarsen.size();

    //cout << "presize: " << presize << endl;

    for(std::list<int>::iterator iter = cells_to_coarsen.begin();
	iter != cells_to_coarsen.end(); iter++)
    {
      // Map cells to new mesh
      if(*iter >= 0)
      {
	*iter = old2new_cell[*iter];
      }
    }

    old2new_cell.resize(mesh.numCells());
    old2new_vertex.resize(mesh.numVertices());

    // Coarsen cells in list
    for(std::list<int>::iterator iter = cells_to_coarsen.begin();
	iter != cells_to_coarsen.end(); iter++)
    {
      bool mesh_ok = false;
      uint cid = *iter;

      if(cid != -1)
      {
	mesh_ok = coarsenCell(mesh, coarse_mesh, cid,
			      old2new_vertex, old2new_cell,
			      coarsen_boundary);
	if(!mesh_ok)
	{
	  cout << "mesh not ok" << endl;
	}
	else
	{
	  mesh = coarse_mesh;
	  cells_to_coarsen.erase(iter);
	  break;
	}
      }
    }

    if(presize == cells_to_coarsen.size())
    {
      break;
    }
  }
}
//-----------------------------------------------------------------------------
void LocalMeshCoarsening::collapseEdge(Mesh& mesh, Edge& edge, 
                                       Vertex& vertex_to_remove, 
                                       MeshFunction<bool>& cell_to_remove, 
                                       Array<int>& old2new_vertex, 
                                       Array<int>& old2new_cell, 
                                       MeshEditor& editor, 
                                       uint& current_cell) 
{
  const CellType& cell_type = mesh.type();
  Array<uint> cell_vertices(cell_type.numEntities(0));

  uint vert_slave = vertex_to_remove.index();
  uint vert_master = 0; 
  uint* edge_vertex = edge.entities(0);
  //cout << "edge vertices: " << edge_vertex[0] << " " << edge_vertex[1] << endl;
  //cout << "vertex: " << vertex_to_remove.index() << endl;

  if ( edge_vertex[0] == vert_slave ) 
    vert_master = edge_vertex[1]; 
  else if ( edge_vertex[1] == vert_slave ) 
    vert_master = edge_vertex[0]; 
  else
    dolfin_error("Node to delete and edge to collapse not compatible.");

  for (CellIterator c(vertex_to_remove); !c.end(); ++c)
  {
    if ( cell_to_remove.get(*c) == false ) 
    {
      uint cv_idx = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
      {  
        if ( v->index() == vert_slave )
	  cell_vertices[cv_idx++] = old2new_vertex[vert_master]; 
        else
	  cell_vertices[cv_idx++] = old2new_vertex[v->index()];
      }
      //cout << "adding new cell" << endl;
      editor.addCell(current_cell++, cell_vertices);

      // Update cell map
      old2new_cell[c->index()] = current_cell - 1;
    }    
  }
  
}
//-----------------------------------------------------------------------------
bool LocalMeshCoarsening::coarsenCell(Mesh& mesh, Mesh& coarse_mesh,
				      int cellid,
				      Array<int>& old2new_vertex,
				      Array<int>& old2new_cell,
				      bool coarsen_boundary)
{
  cout << "coarsenCell: " << cellid << endl;
  cout << "numCells: " << mesh.numCells() << endl;

  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Initialise forbidden verticies   
  MeshFunction<bool> vertex_forbidden(mesh);  
  vertex_forbidden.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden.set(v->index(),false);

  // Initialise boundary verticies   
  MeshFunction<bool> vertex_boundary(mesh);  
  vertex_boundary.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_boundary.set(v->index(),false);

  MeshFunction<uint> bnd_vertex_map; 
  MeshFunction<uint> bnd_cell_map; 
  BoundaryMesh boundary(mesh,bnd_vertex_map,bnd_cell_map);
  for (VertexIterator v(boundary); !v.end(); ++v)
    vertex_boundary.set(bnd_vertex_map.get(v->index()),true);

  // If coarsen boundary is forbidden 
  if ( coarsen_boundary == false )
  {
    for (VertexIterator v(boundary); !v.end(); ++v)
      vertex_forbidden.set(bnd_vertex_map.get(v->index()),true);
  }



  // Initialise data for finding which vertex to remove   
  bool collapse_edge = false;
  uint* edge_vertex;
  uint shortest_edge_index = 0;
  real lmin, l;
  uint num_cells_to_remove = 0;
  
  // Get cell type
  const CellType& cell_type = mesh.type();

  Cell c(mesh, cellid);

  MeshEditor editor;
  editor.open(coarse_mesh, cell_type.cellType(),
	      mesh.topology().dim(), mesh.geometry().dim());

  MeshFunction<bool> cell_to_remove(mesh);  
  cell_to_remove.init(mesh.topology().dim());
  for (CellIterator ci(mesh); !ci.end(); ++ci)
    cell_to_remove.set(ci->index(), false);

  MeshFunction<bool> cell_to_regenerate(mesh);  
  cell_to_regenerate.init(mesh.topology().dim());
  for (CellIterator ci(mesh); !ci.end(); ++ci)
    cell_to_regenerate.set(ci->index(), false);

  // Find shortest edge of cell c
  collapse_edge = false;
  lmin = 1.0e10 * c.diameter();
  for (EdgeIterator e(c); !e.end(); ++e)
  {
    edge_vertex = e->entities(0);
    if ( (vertex_forbidden.get(edge_vertex[0]) == false) || 
	 (vertex_forbidden.get(edge_vertex[1]) == false) )
    {

      l = e->length();
      if ( lmin > l )
      {
	lmin = l;
	shortest_edge_index = e->index(); 
	collapse_edge = true;
      }
    }
  }
  
  Edge shortest_edge(mesh, shortest_edge_index);

  // Decide which vertex to remove
  uint vert2remove_idx = 0;
    
  // If at least one vertex should be removed 
  if ( collapse_edge == true )
  {
    edge_vertex = shortest_edge.entities(0);
    
    if(vertex_forbidden.get(edge_vertex[0]) &&
       vertex_forbidden.get(edge_vertex[1]))
    {
      // Both vertices are forbidden, cannot coarsen

      cout << "both vertices forbidden" << endl;

      editor.close();
      return false;
    }
    if(vertex_forbidden.get(edge_vertex[0]) == true)
    {
      vert2remove_idx = edge_vertex[1];
    }
    else if(vertex_forbidden.get(edge_vertex[1]) == true)
    {
      vert2remove_idx = edge_vertex[0];
    }
    else if ( edge_vertex[0] > edge_vertex[1] ) 
    {
      vert2remove_idx = edge_vertex[0];
    }
    else
    {
      vert2remove_idx = edge_vertex[1];
    }       
    
  }
  else
  {
    // No vertices to remove, cannot coarsen

    cout << "all vertices forbidden" << endl;
    editor.close();
    return false;
  }

  Vertex vertex_to_remove(mesh, vert2remove_idx);

  //cout << "edge vertices2: " << edge_vertex[0] << " " << edge_vertex[1] << endl;
  //cout << "vertex2: " << vertex_to_remove.index() << endl;
  //cout << "collapse: " << collapse_edge << endl;

  // Remove cells around edge 
  num_cells_to_remove = 0;
  for (CellIterator cn(shortest_edge); !cn.end(); ++cn)
  {
    cell_to_remove.set(cn->index(),true);
    num_cells_to_remove++;

    // Update cell map
    old2new_cell[cn->index()] = -1;
  }

  // Regenerate cells around vertex
  for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
  {
    cell_to_regenerate.set(cn->index(),true);

    // Update cell map (will be filled in with correct index)
    old2new_cell[cn->index()] = -1;
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices - 1);
  editor.initCells(num_cells - num_cells_to_remove);
  
  cout << "Number of cells in old mesh: " << num_cells << "; to remove: " <<
    num_cells_to_remove << endl;

  // Add old vertices
  uint vertex = 0;
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    if(vertex_to_remove.index() == v->index()) 
    {
      old2new_vertex[v->index()] = -1;
    }
    else
    {
      old2new_vertex[v->index()] = vertex;
      editor.addVertex(vertex++, v->point());
    }
  }

  // Add old unrefined cells 
  uint cv_idx;
  uint current_cell = 0;
  Array<uint> cell_vertices(cell_type.numEntities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if(cell_to_remove.get(*c) == false && cell_to_regenerate(*c) == false)
    {
      cv_idx = 0;
      for (VertexIterator v(c); !v.end(); ++v)
        cell_vertices[cv_idx++] = old2new_vertex[v->index()]; 
      //cout << "adding old cell" << endl;
      editor.addCell(current_cell++, cell_vertices);

      // Update cell maps
      old2new_cell[c->index()] = current_cell - 1;
    }
  }
  
  // Add new cells. 
  collapseEdge(mesh, shortest_edge, vertex_to_remove, cell_to_remove,
	       old2new_vertex, old2new_cell, editor, current_cell);

  editor.close();

  // Set volume tolerance. This parameter detemines a quality criterion 
  // for the new mesh: higher value indicates a sharper criterion. 
  real vol_tol = 1.0e-3; 

  bool mesh_ok = true;


  // Check mesh quality (volume)
  for (CellIterator c(coarse_mesh); !c.end(); ++c)
  {
    //cout << "c[" << c->index() << "] volume: " << c->volume() << endl;
    real qm = c->volume() / c->diameter();
    if(qm < vol_tol)
    {
      mesh_ok = false;
      //cout << "qm: " << qm << endl;
    }
  }

  // Checking for inverted cells
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    uint id = c->index();
    uint nid = old2new_cell[id];

    if(nid != -1)
    {
      Cell cn(coarse_mesh, nid);

      if(c->orientation() != cn.orientation())
      {
	mesh_ok = false;
      }
    }
  }

  return mesh_ok;

}
//-----------------------------------------------------------------------------

