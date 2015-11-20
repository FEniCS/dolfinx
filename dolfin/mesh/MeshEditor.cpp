// Copyright (C) 2006-2012 Anders Logg
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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2006-05-16
// Last changed: 2014-02-06

#include <dolfin/log/log.h>
#include <dolfin/geometry/Point.h>
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshEditor.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEditor::MeshEditor() : _mesh(0), _tdim(0), _gdim(0), _num_vertices(0),
                           _num_cells(0), next_vertex(0), next_cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEditor::~MeshEditor()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, std::size_t tdim,
                      std::size_t gdim, std::size_t degree)
{
  switch (tdim)
  {
  case 0:
    open(mesh, CellType::point, tdim, gdim, degree);
    break;
  case 1:
    open(mesh, CellType::interval, tdim, gdim, degree);
    break;
  case 2:
    open(mesh, CellType::triangle, tdim, gdim, degree);
    break;
  case 3:
    open(mesh, CellType::tetrahedron, tdim, gdim, degree);
    break;
  default:
    dolfin_error("MeshEditor.cpp",
                 "open mesh for editing",
                 "Unknown cell type of topological dimension %d", tdim);
  }
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, CellType::Type type, std::size_t tdim,
                      std::size_t gdim, std::size_t degree)
{
  // Clear old mesh data
  mesh.clear();
  clear();

  // Save mesh and dimension
  _mesh = &mesh;
  _gdim = gdim;
  _tdim = tdim;

  // Set cell type
  mesh._cell_type.reset(CellType::create(type));

  // Initialize topological dimension
  mesh._topology.init(tdim);

  // Initialize geometric dimension and degree
  mesh._geometry.init(gdim, degree);

  // Initialize domains
  mesh._domains.init(tdim);

  // Initialize temporary storage for local cell data
  _vertices = std::vector<std::size_t>(mesh.type().num_vertices(tdim), 0);
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, std::string type, std::size_t tdim,
                      std::size_t gdim, std::size_t degree)
{
  if (type == "point")
    open(mesh, CellType::point, tdim, gdim, degree);
  else if (type == "interval")
    open(mesh, CellType::interval, tdim, gdim, degree);
  else if (type == "triangle")
    open(mesh, CellType::triangle, tdim, gdim, degree);
  else if (type == "tetrahedron")
    open(mesh, CellType::tetrahedron, tdim, gdim, degree);
  else
  {
    dolfin_error("MeshEditor.cpp",
                 "open mesh for editing",
                 "Unknown cell type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void MeshEditor::init_vertices_global(std::size_t num_local_vertices,
                                      std::size_t num_global_vertices)
{
  // Check if we are currently editing a mesh
  if (!_mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "initialize vertices in mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Initialize mesh data
  _num_vertices = num_local_vertices;
  _mesh->_topology.init(0, num_local_vertices, num_global_vertices);
  _mesh->_topology.init_ghost(0, num_local_vertices);
  _mesh->_topology.init_global_indices(0, num_local_vertices);
  std::vector<std::size_t> num_vertex_points(1, num_local_vertices);
  _mesh->_geometry.init_entities(num_vertex_points);
}
//-----------------------------------------------------------------------------
void MeshEditor::init_entities()
{
  dolfin_assert(_tdim > 0);
  dolfin_assert(_mesh);

  if (_mesh->topology().size(_tdim) == 0
      or next_cell != _num_cells)
  {
    dolfin_error("MeshEditor.cpp",
                 "initialise topological entities",
                 "It is required to add all cells before initialising other entities");
  }

  // Initialise entities required for this polynomial degree
  _mesh->order();
  std::size_t dmax = std::min(_mesh->_geometry.degree(), _tdim);
  for (std::size_t d = 1; d < dmax; ++d)
    _mesh->init(d);

  // Create space in the MeshGeometry coordinates for the new entity points
  std::vector<std::size_t> num_entities(_tdim + 1, 0);
  for (std::size_t d = 0; d <= _tdim; ++d)
    num_entities[d] = _mesh->topology().size(d);
  _mesh->_geometry.init_entities(num_entities);
}
//-----------------------------------------------------------------------------
void MeshEditor::init_cells_global(std::size_t num_local_cells,
                                   std::size_t num_global_cells)
{
  // Check if we are currently editing a mesh
  if (!_mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "initialize cells in mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Initialize mesh data
  _num_cells = num_local_cells;
  _mesh->_topology.init(_tdim, num_local_cells, num_global_cells);
  _mesh->_topology.init_ghost(_tdim, num_local_cells);
  _mesh->_topology.init_global_indices(_tdim, num_local_cells);
  _mesh->_topology(_tdim, 0).init(_num_cells,
                                  _mesh->type().num_vertices(_tdim));
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, const Point& p)
{
  add_vertex_global(index, index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, const std::vector<double>& x)
{
  add_vertex_global(index, index, x);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, double x)
{
  dolfin_assert(_gdim == 1);
  std::vector<double> p(1);
  p[0] = x;
  add_vertex(index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, double x, double y)
{
  dolfin_assert(_gdim == 2);
  std::vector<double> p(2);
  p[0] = x;
  p[1] = y;
  add_vertex(index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, double x, double y, double z)
{
  dolfin_assert(_gdim == 3);
  std::vector<double> p(3);
  p[0] = x;
  p[1] = y;
  p[2] = z;
  add_vertex(index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex_global(std::size_t local_index,
                                   std::size_t global_index,
                                   const Point& p)
{
  // Add vertex
  add_vertex_common(local_index, _gdim);

  // Set coordinate
  _mesh->_geometry.set(local_index, p.coordinates());
  _mesh->_topology.set_global_index(0, local_index, global_index);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex_global(std::size_t local_index,
                                   std::size_t global_index,
                                   const std::vector<double>& x)
{
  // Add vertex
  add_vertex_common(local_index, x.size());

  // Set coordinate
  _mesh->_geometry.set(local_index, x.data());
  _mesh->_topology.set_global_index(0, local_index, global_index);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_entity_point(std::size_t entity_dim, std::size_t order,
                                  std::size_t index, const Point& p)
{
  const std::size_t idx = _mesh->_geometry.get_entity_index(entity_dim, order, index);
  _mesh->_geometry.set(idx, p.coordinates());
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1)
{
  dolfin_assert(_tdim == 1);
  dolfin_assert(_vertices.size() == 2);
  _vertices[0] = v0;
  _vertices[1] = v1;
  add_cell(c, c, _vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                          std::size_t v2)
{
  dolfin_assert(_tdim == 2);
  dolfin_assert(_vertices.size() == 3);
  _vertices[0] = v0;
  _vertices[1] = v1;
  _vertices[2] = v2;
  add_cell(c, c, _vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                          std::size_t v2, std::size_t v3)
{
  dolfin_assert(_tdim == 3);
  dolfin_assert(_vertices.size() == 4);
  _vertices[0] = v0;
  _vertices[1] = v1;
  _vertices[2] = v2;
  _vertices[3] = v3;
  add_cell(c, c, _vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::close(bool order)
{
  // Order mesh if requested
  dolfin_assert(_mesh);
  if (order && !_mesh->ordered())
    _mesh->order();

  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex_common(std::size_t v, std::size_t gdim)
{
  // Check if we are currently editing a mesh
  if (!_mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Check that the dimension matches
  if (gdim != _gdim)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Illegal dimension for vertex coordinate (%d), expecting %d",
                 gdim, _gdim);
  }

  // Check value of vertex index
  if (v >= _num_vertices)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Vertex index (%d) out of range [0, %d)",
                 v, _num_vertices);
  }

  // Check if there is room for more vertices
  if (next_vertex >= _num_vertices)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Vertex list is full, %d vertices already specified",
                 _num_vertices);
  }

  // Step to next vertex
  next_vertex++;
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell_common(std::size_t c, std::size_t tdim)
{
  // Check if we are currently editing a mesh
  if (!_mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Check that the dimension matches
  if (tdim != _tdim)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Illegal dimension for cell (%d), expecting %d", tdim, _tdim);
  }

  // Check value of cell index
  if (c >= _num_cells)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Cell index (%d) out of range [0, %d)",
                 c, _num_cells);
  }

  // Check if there is room for more cells
  if (next_cell >= _num_cells)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Cell list is full, %d cells already specified",
                 _num_cells);
  }

  // Step to next cell
  next_cell++;
}
//-----------------------------------------------------------------------------
void MeshEditor::clear()
{
  _tdim = 0;
  _gdim = 0;
  _num_vertices = 0;
  _num_cells = 0;
  next_vertex = 0;
  next_cell = 0;
  _mesh = 0;
  _vertices.clear();
}
//-----------------------------------------------------------------------------
