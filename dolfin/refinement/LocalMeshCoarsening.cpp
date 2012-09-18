// Copyright (C) 2006 Johan Hoffman
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2008.
//
// First added:  2006-11-01
// Last changed: 2011-04-07

#include <list>

#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/TetrahedronCell.h>
#include <dolfin/mesh/TriangleCell.h>
#include <dolfin/mesh/Vertex.h>
#include "LocalMeshCoarsening.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalMeshCoarsening::coarsen_mesh_by_edge_collapse(Mesh& mesh,
                                                        MeshFunction<bool>& cell_marker,
                                                        bool coarsen_boundary)
{
  log(TRACE, "Coarsen simplicial mesh by edge collapse.");

  // Get size of old mesh
  //const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Check cell marker
  if ( cell_marker.size() != num_cells )
    dolfin_error("LocalMeshCoarsening.cpp",
                 "coarsen mesh by collapsing edges",
                 "Number of cell markers (%d) does not match number of cells (%d)",
                 cell_marker.size(), num_cells);

  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);

  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);

  // Get cell type
  //const CellType& cell_type = mesh.type();

  // Create new mesh
  Mesh coarse_mesh(mesh);

  // Initialise forbidden cells
  MeshFunction<bool> cell_forbidden(mesh);
  cell_forbidden.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_forbidden[c->index()] = false;

  // Init new vertices and cells
  std::vector<int> old2new_cell(mesh.num_cells());
  std::vector<int> old2new_vertex(mesh.num_vertices());

  std::list<int> cells_to_coarsen(0);

  // Compute cells to delete
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (cell_marker[*c] == true)
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

    uint presize = cells_to_coarsen.size();

    //cout << "presize: " << presize << endl;

    for(std::list<int>::iterator iter = cells_to_coarsen.begin();
          iter != cells_to_coarsen.end(); iter++)
    {
      // Map cells to new mesh
      if(*iter >= 0)
        *iter = old2new_cell[*iter];
    }

    old2new_cell.resize(mesh.num_cells());
    old2new_vertex.resize(mesh.num_vertices());

    // Coarsen cells in list
    for(std::list<int>::iterator iter = cells_to_coarsen.begin();
          iter != cells_to_coarsen.end(); iter++)
    {
      bool mesh_ok = false;
      int cid = *iter;
      if(cid != -1)
      {
        mesh_ok = coarsen_cell(mesh, coarse_mesh, cid,
			                        old2new_vertex, old2new_cell,
			                        coarsen_boundary);
        if(!mesh_ok)
          warning("Mesh not ok");
        else
        {
          mesh = coarse_mesh;
          cells_to_coarsen.erase(iter);
          break;
        }
      }
    }

    if(presize == cells_to_coarsen.size())
      break;
  }
}
//-----------------------------------------------------------------------------
void LocalMeshCoarsening::collapse_edge(Mesh& mesh, Edge& edge,
                                       Vertex& vertex_to_remove,
                                       MeshFunction<bool>& cell_to_remove,
                                       std::vector<int>& old2new_vertex,
                                       std::vector<int>& old2new_cell,
                                       MeshEditor& editor,
                                       uint& current_cell)
{
  const CellType& cell_type = mesh.type();
  std::vector<uint> cell_vertices(cell_type.num_entities(0));

  uint vert_slave = vertex_to_remove.index();
  uint vert_master = 0;
  const uint* edge_vertex = edge.entities(0);
  //cout << "edge vertices: " << edge_vertex[0] << " " << edge_vertex[1] << endl;
  //cout << "vertex: " << vertex_to_remove.index() << endl;

  if ( edge_vertex[0] == vert_slave )
    vert_master = edge_vertex[1];
  else if ( edge_vertex[1] == vert_slave )
    vert_master = edge_vertex[0];
  else
    dolfin_error("LocalMeshCoarsening.cpp",
                 "collapse edge",
                 "The node to delete and edge to collapse are not compatible");

  for (CellIterator c(vertex_to_remove); !c.end(); ++c)
  {
    if ( cell_to_remove[*c] == false )
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
      editor.add_cell(current_cell++, cell_vertices);

      // Update cell map
      old2new_cell[c->index()] = current_cell - 1;
    }
  }

}
//-----------------------------------------------------------------------------
bool LocalMeshCoarsening::coarsen_cell(Mesh& mesh, Mesh& coarse_mesh,
				      int cellid,
				      std::vector<int>& old2new_vertex,
				      std::vector<int>& old2new_cell,
				      bool coarsen_boundary)
{
  cout << "coarsen_cell: " << cellid << endl;
  cout << "num_cells: " << mesh.num_cells() << endl;

  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Initialise forbidden verticies
  MeshFunction<bool> vertex_forbidden(mesh);
  vertex_forbidden.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_forbidden[v->index()] = false;

  // Initialise boundary verticies
  MeshFunction<bool> vertex_boundary(mesh);
  vertex_boundary.init(0);
  for (VertexIterator v(mesh); !v.end(); ++v)
    vertex_boundary[v->index()] = false;

  BoundaryMesh boundary(mesh);
  MeshFunction<unsigned int>& bnd_vertex_map = boundary.vertex_map();
  for (VertexIterator v(boundary); !v.end(); ++v)
    vertex_boundary[bnd_vertex_map[v->index()]] = true;

  // If coarsen boundary is forbidden
  if (coarsen_boundary == false)
  {
    for (VertexIterator v(boundary); !v.end(); ++v)
      vertex_forbidden[bnd_vertex_map[v->index()]] = true;
  }
  // Initialise data for finding which vertex to remove
  bool _collapse_edge = false;
  const uint* edge_vertex;
  uint shortest_edge_index = 0;
  double lmin, l;
  uint num_cells_to_remove = 0;

  // Get cell type
  const CellType& cell_type = mesh.type();
  const Cell c(mesh, cellid);

  MeshEditor editor;
  editor.open(coarse_mesh, cell_type.cell_type(),
              mesh.topology().dim(), mesh.geometry().dim());

  MeshFunction<bool> cell_to_remove(mesh);
  cell_to_remove.init(mesh.topology().dim());

  for (CellIterator ci(mesh); !ci.end(); ++ci)
    cell_to_remove[ci->index()] = false;

  MeshFunction<bool> cell_to_regenerate(mesh);
  cell_to_regenerate.init(mesh.topology().dim());
  for (CellIterator ci(mesh); !ci.end(); ++ci)
    cell_to_regenerate[ci->index()] = false;

  // Find shortest edge of cell c
  _collapse_edge = false;
  lmin = 1.0e10 * c.diameter();
  for (EdgeIterator e(c); !e.end(); ++e)
  {
    edge_vertex = e->entities(0);
    if (!vertex_forbidden[edge_vertex[0]] || !vertex_forbidden[edge_vertex[1]])
    {
      l = e->length();
      if ( lmin > l )
      {
        lmin = l;
        shortest_edge_index = e->index();
        _collapse_edge = true;
      }
    }
  }

  Edge shortest_edge(mesh, shortest_edge_index);

  // Decide which vertex to remove
  uint vert2remove_idx = 0;

  // If at least one vertex should be removed
  if ( _collapse_edge == true )
  {
    edge_vertex = shortest_edge.entities(0);

    if(vertex_forbidden[edge_vertex[0]] &&
       vertex_forbidden[edge_vertex[1]])
    {
      // Both vertices are forbidden, cannot coarsen

      cout << "both vertices forbidden" << endl;

      editor.close();
      return false;
    }
    if(vertex_forbidden[edge_vertex[0]] == true)
      vert2remove_idx = edge_vertex[1];
    else if(vertex_forbidden[edge_vertex[1]] == true)
      vert2remove_idx = edge_vertex[0];
    else if(vertex_boundary[edge_vertex[1]] == true && vertex_boundary[edge_vertex[0]] == false)
      vert2remove_idx = edge_vertex[0];
    else if(vertex_boundary[edge_vertex[0]] == true && vertex_boundary[edge_vertex[1]] == false)
      vert2remove_idx = edge_vertex[1];
    else if ( edge_vertex[0] > edge_vertex[1] )
      vert2remove_idx = edge_vertex[0];
    else
      vert2remove_idx = edge_vertex[1];
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
  //cout << "collapse: " << _collapse_edge << endl;

  // Remove cells around edge
  num_cells_to_remove = 0;
  for (CellIterator cn(shortest_edge); !cn.end(); ++cn)
  {
    cell_to_remove[cn->index()] = true;
    num_cells_to_remove++;

    // Update cell map
    old2new_cell[cn->index()] = -1;
  }

  // Regenerate cells around vertex
  for (CellIterator cn(vertex_to_remove); !cn.end(); ++cn)
  {
    cell_to_regenerate[cn->index()] = true;

    // Update cell map (will be filled in with correct index)
    old2new_cell[cn->index()] = -1;
  }

  // Specify number of vertices and cells
  editor.init_vertices(num_vertices - 1);
  editor.init_cells(num_cells - num_cells_to_remove);

  cout << "Number of cells in old mesh: " << num_cells << "; to remove: " <<
    num_cells_to_remove << endl;

  // Add old vertices
  uint vertex = 0;
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    if(vertex_to_remove.index() == v->index())
      old2new_vertex[v->index()] = -1;
    else
    {
      //cout << "adding old vertex at: " << v->point() << endl;
      old2new_vertex[v->index()] = vertex;
      editor.add_vertex(vertex, vertex, v->point());
      vertex++;
    }
  }

  // Add old unrefined cells
  uint cv_idx;
  uint current_cell = 0;
  std::vector<uint> cell_vertices(cell_type.num_entities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if(cell_to_remove[*c] == false && cell_to_regenerate[*c] == false)
    {
      cv_idx = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
        cell_vertices[cv_idx++] = old2new_vertex[v->index()];
      //cout << "adding old cell" << endl;
      editor.add_cell(current_cell++, cell_vertices);

      // Update cell maps
      old2new_cell[c->index()] = current_cell - 1;
    }
  }

  // Add new cells.
  collapse_edge(mesh, shortest_edge, vertex_to_remove, cell_to_remove,
                old2new_vertex, old2new_cell, editor, current_cell);

  editor.close();

  // Set volume tolerance. This parameter detemines a quality criterion
  // for the new mesh: higher value indicates a sharper criterion.
  double vol_tol = 1.0e-3;

  bool mesh_ok = true;

  Cell removed_cell(mesh, cellid);

  // Check mesh quality (volume)
  for (CellIterator c(removed_cell); !c.end(); ++c)
  {
    uint id = c->index();
    int nid = old2new_cell[id];

    if(nid != -1)
    {
      Cell cn(coarse_mesh, nid);
      double qm = cn.volume() / cn.diameter();
      if(qm < vol_tol)
      {
        warning("Cell quality too low");
        cout << "qm: " << qm << endl;
        mesh_ok = false;
        return mesh_ok;
      }
    }
  }

  // Checking for inverted cells
  for (CellIterator c(removed_cell); !c.end(); ++c)
  {
    uint id = c->index();
    int nid = old2new_cell[id];

    if(nid != -1)
    {
      Cell cn(coarse_mesh, nid);

      if(c->orientation() != cn.orientation())
      {
        cout << "cell orientation inverted" << endl;
        mesh_ok = false;
        return mesh_ok;
      }
    }
  }
  return mesh_ok;
}
//-----------------------------------------------------------------------------
