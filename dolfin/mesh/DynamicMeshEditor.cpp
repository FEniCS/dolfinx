// Copyright (C) 2008-2011 Anders Logg
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
// First added:  2008-09-22
// Last changed: 2014-02-06

#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include "Mesh.h"
#include "MeshEditor.h"
#include "DynamicMeshEditor.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DynamicMeshEditor::DynamicMeshEditor() : _mesh(0), _tdim(0), _gdim(0),
                                         _cell_type(0),
                                         _num_global_vertices(0),
                                         _num_global_cells(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DynamicMeshEditor::~DynamicMeshEditor()
{
  clear();
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::open(Mesh& mesh, CellType::Type type, std::size_t tdim,
                             std::size_t gdim, std::size_t num_global_vertices,
                             std::size_t num_global_cells)
{
  // Clear old data
  clear();

  // Save data
  _mesh = &mesh;
  _gdim = gdim;
  _tdim = tdim;
  _cell_type = CellType::create(type);
  _num_global_vertices = num_global_vertices;
  _num_global_cells    = num_global_cells;
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::open(Mesh& mesh, std::string type, std::size_t tdim,
                             std::size_t gdim, std::size_t num_global_vertices,
                             std::size_t num_global_cells)
{
  if (type == "point")
  {
    open(mesh, CellType::point, tdim, gdim, num_global_vertices,
         num_global_cells);
  }
  else if (type == "interval")
  {
    open(mesh, CellType::interval, tdim, gdim, num_global_vertices,
         num_global_cells);
  }
  else if (type == "triangle")
  {
    open(mesh, CellType::triangle, tdim, gdim, num_global_vertices,
         num_global_cells);
  }
  else if (type == "tetrahedron")
  {
    open(mesh, CellType::tetrahedron, tdim, gdim, num_global_vertices,
         num_global_cells);
  }
  else
  {
    dolfin_error("DynamicMeshEditor.cpp",
                 "open dynamic mesh editor",
                 "Unknown cell type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(std::size_t v, const Point& p)
{
  // Resize array if necessary
  const std::size_t offset = v*_gdim;
  const std::size_t size = offset + _gdim;
  if (size > vertex_coordinates.size())
    vertex_coordinates.resize(size, 0.0);

  // Set coordinates
  for (std::size_t i = 0; i < _gdim; i++)
    vertex_coordinates[offset + i] = p[i];
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(std::size_t v, double x)
{
  Point p(x);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(std::size_t v, double x, double y)
{
  Point p(x, y);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(std::size_t v, double x, double y, double z)
{
  Point p(x, y, z);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(std::size_t c,
                                 const std::vector<std::size_t>& v)
{
  // Check size of array
  const std::size_t vertices_per_cell = _cell_type->num_vertices(_tdim);
  if (v.size() != vertices_per_cell)
  {
    dolfin_error("DynamicMeshEditor.cpp",
                 "add cell using dynamic mesh editor",
                 "Illegal number of vertices (%d) for cell, expected %d",
                 v.size(), vertices_per_cell);
  }

  // Resize array if necessary
  const std::size_t offset = c*vertices_per_cell;
  const std::size_t size = offset + vertices_per_cell;
  if (size > cell_vertices.size())
    cell_vertices.resize(size, 0);

  // Set vertices
  for (std::size_t i = 0; i < vertices_per_cell; i++)
    cell_vertices[offset + i] = v[i];
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1)
{
  std::vector<std::size_t> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(std::size_t c, std::size_t v0,
                                 std::size_t v1, std::size_t v2)
{
  std::vector<std::size_t> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  vertices.push_back(v2);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(std::size_t c, std::size_t v0,
                                 std::size_t v1, std::size_t v2,
                                 std::size_t v3)
{
  std::vector<std::size_t> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  vertices.push_back(v2);
  vertices.push_back(v3);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::close(bool order)
{
  dolfin_assert(_mesh);
  dolfin_assert(_cell_type);

  // Open default mesh editor
  MeshEditor editor;
  editor.open(*_mesh, _cell_type->cell_type(), _tdim, _gdim);

  // Set number of vertices
  const std::size_t num_vertices = vertex_coordinates.size()/_gdim;
  editor.init_vertices_global(num_vertices, _num_global_vertices);

  // Set number of cells
  const std::size_t vertices_per_cell = _cell_type->num_vertices(_gdim);
  const std::size_t num_cells = cell_vertices.size()/vertices_per_cell;
  editor.init_cells_global(num_cells, _num_global_cells);

  // Add vertices
  std::vector<double> p(_gdim);
  for (std::size_t v = 0; v < num_vertices; v++)
  {
    const std::size_t offset = v*_gdim;
    for (std::size_t i = 0; i < _gdim; i++)
      p[i] = vertex_coordinates[offset + i];
    editor.add_vertex(v, p);
  }

  // Add cells
  std::vector<std::size_t> vertices(vertices_per_cell);
  for (std::size_t c = 0; c < num_cells; c++)
  {
    const std::size_t offset = c*vertices_per_cell;
    for (std::size_t i = 0; i < vertices_per_cell; i++)
      vertices[i] = cell_vertices[offset + i];
    editor.add_cell(c, vertices);
  }

  // Close editor
  editor.close(order);

  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::clear()
{
  _mesh = 0;
  _tdim = 0;
  _gdim = 0;

  delete _cell_type;
  _cell_type = 0;

  vertex_coordinates.clear();
  cell_vertices.clear();
}
//-----------------------------------------------------------------------------
