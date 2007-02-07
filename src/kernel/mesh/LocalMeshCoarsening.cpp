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

  // Initialise forbidden cells 
  MeshFunction<bool> cell_forbidden(mesh);  
  cell_forbidden.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);
  
  // Initialise data for finding which vertex to remove   
  bool collapse_edge = false;
  uint* edge_vertex;
  uint shortest_edge_index = 0;
  real lmin, l;

  // Compute number of vertices and cells 
  for (CellIterator c(mesh); !c.end(); ++c)
  {

    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {

      // Find shortest edge of cell c
      collapse_edge = false;
      lmin = 1.0e10 * c->diameter();
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        edge_vertex = e->entities(0);
        if ( (vertex_forbidden.get(edge_vertex[0]) == false) || 
             (vertex_forbidden.get(edge_vertex[1]) == false) )
        {
          l = e->length();
          if ( lmin > l )
          {
            if ( collapseEdgeOk(mesh,e->index(),edge_vertex,vertex_forbidden) ) 
            {
              lmin = l;
              shortest_edge_index = e->index(); 
              collapse_edge = true;
            }
          }
        }
      }

      // If at least one vertex should be removed 
      if ( collapse_edge == true )
      {
        Edge shortest_edge(mesh,shortest_edge_index);

        uint vert2remove_idx = 0;
        edge_vertex = shortest_edge.entities(0);
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

        for (VertexIterator v(vertex_to_remove); !v.end(); ++v)
          vertex_forbidden.set(v->index(),true);
        
	for (CellIterator cn(shortest_edge); !cn.end(); ++cn)
	{
          // remove cell
          //          if ( cell_forbidden.get(cn->index()) == false )
          // {          
            cell_to_remove.set(cn->index(),true);
            num_cells_to_remove++;
            //}

	}

        // Set cells of vertex to remove to forbidden 
        for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
          cell_forbidden.set(cn->index(),true);
      }
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_vertices - num_vertices_to_remove);
  editor.initCells(num_cells - num_cells_to_remove);
  
  cout << "Number of cells in old mesh: " << num_cells << "; to remove: " << num_cells_to_remove << endl;
  cout << "Number of vertices in old mesh: " << num_vertices << "; to remove: " << num_vertices_to_remove << endl;
  
  // Add old vertices
  Array<int> old2new_vertex(num_vertices);
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    if ( vertex_to_remove_index.get(*v) == false ) 
    {
      old2new_vertex[v->index()] = vertex;
      editor.addVertex(vertex++, v->point());
    }
    else
    {
      old2new_vertex[v->index()] = -1;
    }
  }

  //???????????????
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_to_remove_index.set(v->index(),false);
  //???????????????


  // Add old unrefined cells 
  uint cv_idx;
  uint current_cell = 0;
  Array<uint> cell_vertices(cell_type.numEntities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( cell_forbidden.get(*c) == false )
    {
      cv_idx = 0;
      for (VertexIterator v(c); !v.end(); ++v)
        cell_vertices[cv_idx++] = old2new_vertex[v->index()]; 
      editor.addCell(current_cell++, cell_vertices);
    }
  }
  
  // Reset forbidden verticies 
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden.set(v->index(),false);

  // If coarsen boundary is forbidden
  if ( coarsen_boundary == false )
  {
    for (VertexIterator v(boundary); !v.end(); ++v)
      vertex_forbidden.set(bnd_vertex_map.get(v->index()),true);
  }

  // Reset forbidden cells 
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden.set(c->index(),false);

  // Add new vertices and cells. 
  for (CellIterator c(mesh); !c.end(); ++c)
  {

    if ( (cell_marker.get(*c) == true) && (cell_forbidden.get(*c) == false) )
    {

      // Find shortest edge of cell c
      collapse_edge = false;
      lmin = 1.0e10 * c->diameter();
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        edge_vertex = e->entities(0);
        if ( (vertex_forbidden.get(edge_vertex[0]) == false) || 
             (vertex_forbidden.get(edge_vertex[1]) == false) )
        {
          l = e->length();
          if ( lmin > l )
          {
            if ( collapseEdgeOk(mesh,e->index(),edge_vertex,vertex_forbidden) ) 
            {
              lmin = l;
              shortest_edge_index = e->index(); 
              collapse_edge = true;
            }
          }
        }
      }

      // If at least one vertex should be removed 
      if ( collapse_edge == true )
      {
        Edge shortest_edge(mesh,shortest_edge_index);

        uint vert2remove_idx = 0;
        edge_vertex = shortest_edge.entities(0);
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
        collapseEdge(mesh, shortest_edge, vertex_to_remove, cell_to_remove, old2new_vertex, editor, current_cell);

        for (VertexIterator v(vertex_to_remove); !v.end(); ++v)
          vertex_forbidden.set(v->index(),true);

        // Set cells of vertex to remove to forbidden 
        for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
          cell_forbidden.set(cn->index(),true);

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
                                       Array<int>& old2new_vertex, 
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
  else if ( edge_vertex[1] == vert_slave ) 
    vert_master = edge_vertex[0]; 
  else
    dolfin_error("Node to delte and edge to collapse not compatible.");

  for (CellIterator c(vertex_to_remove); !c.end(); ++c)
  {
    if ( cell_to_remove.get(*c) == false ) 
    {
      uint cv_idx = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
      {  
        if ( v->index() == vert_slave ) cell_vertices[cv_idx++] = old2new_vertex[vert_master]; 
        else                            cell_vertices[cv_idx++] = old2new_vertex[v->index()];
      }
      editor.addCell(current_cell++, cell_vertices);
    }    
  }
  
}
//-----------------------------------------------------------------------------
bool LocalMeshCoarsening::collapseEdgeOk(Mesh& mesh, 
                                         uint edge_index,
                                         uint* edge_vertex,
                                         MeshFunction<bool>& vertex_forbidden)  
{
  // Create vertices 
  Vertex v0(mesh,edge_vertex[0]);
  Vertex v1(mesh,edge_vertex[1]);

  Edge edge(mesh,edge_index);

  // Get mesh geometry
  MeshGeometry& geometry = mesh.geometry(); 
  
  // Set volume tolerance. This parameter detemines a quality criterion 
  // for the new mesh: higher value indicates a sharper criterion. 
  real vol_tol_wt = 0.1; 
  
  if ( !vertex_forbidden(v0) ) 
  {
    for (CellIterator c(v0); !c.end(); ++c)
    {  
      real vol_tol = vol_tol_wt * c->volume();
      
      // Get cell type
      CellType::Type cell_type = mesh.type().cellType();
      
      uint* vertices = c->entities(0);

      real v = 0.0;
      switch ( cell_type )
      {
      case CellType::interval:
      { 
        dolfin_error("Local mesh coarsening not implemented for an interval.");
      }
      case CellType::triangle:
      {
        real* x0 = geometry.x(vertices[0]);
        real* x1 = geometry.x(vertices[1]);
        real* x2 = geometry.x(vertices[2]);
        
        if ( v0.index() == vertices[0] ) x0 = geometry.x(v1.index());
        if ( v0.index() == vertices[1] ) x1 = geometry.x(v1.index());
        if ( v0.index() == vertices[2] ) x2 = geometry.x(v1.index());
        
        // Formula for triangle area
        /*
          v = ( (x0[0]*x1[1] +  x2[0]*x0[1] +  x1[0]*x2[1]) - 
          (x1[0]*x0[1] +  x0[0]*x2[1] +  x2[0]*x1[1]) );  
        */     
        v = ( (x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0]) );
        v /= 2.0;
        break;
      }
      case CellType::tetrahedron:
      {
        real* x0 = geometry.x(vertices[0]);
        real* x1 = geometry.x(vertices[1]);
        real* x2 = geometry.x(vertices[2]);
        real* x3 = geometry.x(vertices[3]);
        
        if ( v0.index() == vertices[0] ) x0 = geometry.x(v1.index());
        if ( v0.index() == vertices[1] ) x1 = geometry.x(v1.index());
        if ( v0.index() == vertices[2] ) x2 = geometry.x(v1.index());
        if ( v0.index() == vertices[3] ) x3 = geometry.x(v1.index());
        
        // Formula for tetrahedron volume from http://mathworld.wolfram.com
        v = ( x0[0] * ( x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2] - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2] ) -
              x1[0] * ( x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2] - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2] ) +
              x2[0] * ( x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2] - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2] ) -
              x3[0] * ( x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2] - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2] ) );          
        v /= 6.0;
        break;
      }
      default:
        dolfin_error1("Unknown cell type: %d.", cell_type);
      }
 
      bool collapse_cell = false;
      for ( CellIterator ce(edge); !ce.end(); ++ce )
        if ( c->index() == ce->index() ) 
          collapse_cell = true;

      if ( (fabs(v) < vol_tol) && (!collapse_cell) ) vertex_forbidden(v0) = true;

    }
  }

  if ( !vertex_forbidden(v1) ) 
  {
    for (CellIterator c(v1); !c.end(); ++c)
    {  
      real vol_tol = vol_tol_wt * c->volume();
      
      // Get cell type
      CellType::Type cell_type = mesh.type().cellType();
      
      uint* vertices = c->entities(0);

      real v = 0.0;
      switch ( cell_type )
      {
      case CellType::interval:
      { 
        dolfin_error("Local mesh coarsening not implemented for an interval.");
      }
      case CellType::triangle:
      {
        real* x0 = geometry.x(vertices[0]);
        real* x1 = geometry.x(vertices[1]);
        real* x2 = geometry.x(vertices[2]);
        
        if ( v0.index() == vertices[0] ) x0 = geometry.x(v1.index());
        if ( v0.index() == vertices[1] ) x1 = geometry.x(v1.index());
        if ( v0.index() == vertices[2] ) x2 = geometry.x(v1.index());
        
        // Formula for triangle area
        /*
          v = ( (x0[0]*x1[1] +  x2[0]*x0[1] +  x1[0]*x2[1]) - 
          (x1[0]*x0[1] +  x0[0]*x2[1] +  x2[0]*x1[1]) );  
        */     
        v = ( (x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0]) );
        v /= 2.0;
        break;
      }
      case CellType::tetrahedron:
      {
        real* x0 = geometry.x(vertices[0]);
        real* x1 = geometry.x(vertices[1]);
        real* x2 = geometry.x(vertices[2]);
        real* x3 = geometry.x(vertices[3]);
        
        if ( v0.index() == vertices[0] ) x0 = geometry.x(v1.index());
        if ( v0.index() == vertices[1] ) x1 = geometry.x(v1.index());
        if ( v0.index() == vertices[2] ) x2 = geometry.x(v1.index());
        if ( v0.index() == vertices[3] ) x3 = geometry.x(v1.index());
        
        // Formula for tetrahedron volume from http://mathworld.wolfram.com
        v = ( x0[0] * ( x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2] - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2] ) -
              x1[0] * ( x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2] - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2] ) +
              x2[0] * ( x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2] - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2] ) -
              x3[0] * ( x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2] - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2] ) );          
        v /= 6.0;
        break;
      }
      default:
        dolfin_error1("Unknown cell type: %d.", cell_type);
      }
 
      bool collapse_cell = false;
      for ( CellIterator ce(edge); !ce.end(); ++ce )
        if ( c->index() == ce->index() ) 
          collapse_cell = true;

      if ( (fabs(v) < vol_tol) && (!collapse_cell) ) vertex_forbidden(v1) = true;

    }
  }

  if ( (vertex_forbidden(v0) == false) || (vertex_forbidden(v1) == false) ) 
    return true;
  else 
    return false; 
  
}
//-----------------------------------------------------------------------------

