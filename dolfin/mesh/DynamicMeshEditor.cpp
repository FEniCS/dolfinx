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
// Last changed: 2011-11-14

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/dolfin_parameter.h>
#include "Mesh.h"
#include "Point.h"
#include "MeshEditor.h"
#include "DynamicMeshEditor.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DynamicMeshEditor::DynamicMeshEditor()
  : mesh(0), tdim(0), gdim(0), cell_type(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DynamicMeshEditor::~DynamicMeshEditor()
{
  clear();
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim)
{
  // Clear old data
  mesh.clear();
  clear();

  // Save data
  this->mesh = &mesh;
  this->gdim = gdim;
  this->tdim = tdim;
  this->cell_type = CellType::create(type);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::open(Mesh& mesh, std::string type, uint tdim, uint gdim)
{
  if (type == "point")
    open(mesh, CellType::point, tdim, gdim);
  else if (type == "interval")
    open(mesh, CellType::interval, tdim, gdim);
  else if (type == "triangle")
    open(mesh, CellType::triangle, tdim, gdim);
  else if (type == "tetrahedron")
    open(mesh, CellType::tetrahedron, tdim, gdim);
  else
  {
    dolfin_error("DynamicMeshEditor.cpp",
                 "open dynamic mesh editor",
                 "Unknown cell type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(uint v, const Point& p)
{
  // Resize array if necessary
  const uint offset = v*gdim;
  const uint size = offset + gdim;
  if (size > vertex_coordinates.size())
    vertex_coordinates.resize(size, 0.0);

  // Set coordinates
  for (uint i = 0; i < gdim; i++)
    vertex_coordinates[offset + i] = p[i];
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(uint v, double x)
{
  Point p(x);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(uint v, double x, double y)
{
  Point p(x, y);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_vertex(uint v, double x, double y, double z)
{
  Point p(x, y, z);
  add_vertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(uint c, const std::vector<uint>& v)
{
  // Check size of array
  const uint vertices_per_cell = cell_type->num_vertices(tdim);
  if (v.size() != vertices_per_cell)
  {
    dolfin_error("DynamicMeshEditor.cpp",
                 "add cell using dynamic mesh editor",
                 "Illegal number of vertices (%d) for cell, expected %d",
                 v.size(), vertices_per_cell);
  }

  // Resize array if necessary
  const uint offset = c*vertices_per_cell;
  const uint size = offset + vertices_per_cell;
  if (size > cell_vertices.size())
    cell_vertices.resize(size, 0);

  // Set vertices
  for (uint i = 0; i < vertices_per_cell; i++)
    cell_vertices[offset + i] = v[i];
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(uint c, uint v0, uint v1)
{
  std::vector<uint> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(uint c, uint v0, uint v1, uint v2)
{
  std::vector<uint> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  vertices.push_back(v2);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::add_cell(uint c, uint v0, uint v1, uint v2, uint v3)
{
  std::vector<uint> vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  vertices.push_back(v2);
  vertices.push_back(v3);
  add_cell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::close(bool order)
{
  assert(mesh);
  assert(cell_type);

  // Open default mesh editor
  MeshEditor editor;
  editor.open(*mesh, cell_type->cell_type(), tdim, gdim);

  // Set number of vertices
  const uint num_vertices = vertex_coordinates.size() / gdim;
  editor.init_vertices(num_vertices);

  // Set number of cells
  const uint vertices_per_cell = cell_type->num_vertices(gdim);
  const uint num_cells = cell_vertices.size() / vertices_per_cell;
  editor.init_cells(num_cells);

  // Add vertices
  Point p;
  for (uint v = 0; v < num_vertices; v++)
  {
    const uint offset = v*gdim;
    for (uint i = 0; i < gdim; i++)
      p[i] = vertex_coordinates[offset + i];
    editor.add_vertex(v, p);
  }

  // Add cells
  std::vector<uint> vertices(vertices_per_cell);
  for (uint c = 0; c < num_cells; c++)
  {
    const uint offset = c*vertices_per_cell;
    for (uint i = 0; i < vertices_per_cell; i++)
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
  mesh = 0;
  tdim = 0;
  gdim = 0;

  delete cell_type;
  cell_type = 0;

  vertex_coordinates.clear();
  cell_vertices.clear();
}
//-----------------------------------------------------------------------------
