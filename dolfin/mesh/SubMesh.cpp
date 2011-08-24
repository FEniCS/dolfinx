// Copyright (C) 2009-2011 Anders Logg
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
// First added:  2009-02-11
// Last changed: 2011-08-23

#include <map>
#include <vector>

#include "Cell.h"
#include "Vertex.h"
#include "MeshEditor.h"
#include "SubDomain.h"
#include "SubMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh, const SubDomain& sub_domain)
{
  // Create mesh function and mark sub domain
  MeshFunction<uint> sub_domains(mesh, mesh.topology().dim());
  sub_domains = 0;
  sub_domain.mark(sub_domains, 1);

  // Create sub mesh
  init(mesh, sub_domains, 1);
}
//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh,
                 const MeshFunction<uint>& sub_domains, uint sub_domain)
{
  // Create sub mesh
  init(mesh, sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
SubMesh::~SubMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SubMesh::init(const Mesh& mesh,
                   const MeshFunction<uint>& sub_domains, uint sub_domain)
{
  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, mesh.type().cell_type(),
              mesh.topology().dim(), mesh.geometry().dim());

  // Extract cells
  std::set<uint> cells;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (sub_domains[*cell] == sub_domain)
      cells.insert(cell->index());
  }

  // Map to keep track of new local indices for vertices
  std::map<uint, uint> local_vertex_indices;

  // Add cells
  editor.init_cells(cells.size());
  uint current_cell = 0;
  uint current_vertex = 0;
  for (std::set<uint>::iterator cell_it = cells.begin();
       cell_it != cells.end(); ++cell_it)
  {
    std::vector<uint> cell_vertices;
    Cell cell(mesh, *cell_it);
    for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
    {
      const uint parent_vertex_index = vertex->index();
      uint local_vertex_index = 0;
      std::map<uint, uint>::iterator vertex_it
        = local_vertex_indices.find(parent_vertex_index);
      if (vertex_it != local_vertex_indices.end())
      {
        local_vertex_index = vertex_it->second;
      }
      else
      {
        local_vertex_index = current_vertex++;
        local_vertex_indices[parent_vertex_index] = local_vertex_index;
      }
      cell_vertices.push_back(local_vertex_index);
    }
    editor.add_cell(current_cell++, cell_vertices);
  }

  // Add vertices
  editor.init_vertices(local_vertex_indices.size());
  for (std::map<uint, uint>::iterator it = local_vertex_indices.begin();
       it != local_vertex_indices.end(); ++it)
  {
    Vertex vertex(mesh, it->first);
    editor.add_vertex(it->second, vertex.point());
  }

  // Close editor
  editor.close();

  // Build local-to-parent mapping for vertices
  boost::shared_ptr<MeshFunction<unsigned int> > parent_vertex_indices
    = data().create_mesh_function("parent_vertex_indices", 0);
  for (std::map<uint, uint>::iterator it = local_vertex_indices.begin();
       it != local_vertex_indices.end(); ++it)
    (*parent_vertex_indices)[it->second] = it->first;
}
//-----------------------------------------------------------------------------
