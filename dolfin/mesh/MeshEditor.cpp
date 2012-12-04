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
// Last changed: 2012-10-30

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/dolfin_parameter.h>
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "Point.h"
#include "MeshEditor.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEditor::MeshEditor() : mesh(0), tdim(0), gdim(0), num_vertices(0),
                           num_cells(0), next_vertex(0), next_cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEditor::~MeshEditor()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, std::size_t tdim, std::size_t gdim)
{
  switch (tdim)
  {
  case 0:
    open(mesh, CellType::point, tdim, gdim);
    break;
  case 1:
    open(mesh, CellType::interval, tdim, gdim);
    break;
  case 2:
    open(mesh, CellType::triangle, tdim, gdim);
    break;
  case 3:
    open(mesh, CellType::tetrahedron, tdim, gdim);
    break;
  default:
    dolfin_error("MeshEditor.cpp",
                 "open mesh for editing",
                 "Uknown cell type of topological dimension %d", tdim);
  }
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, CellType::Type type, std::size_t tdim, std::size_t gdim)
{
  // Clear old mesh data
  mesh.clear();
  clear();

  // Save mesh and dimension
  this->mesh = &mesh;
  this->gdim = gdim;
  this->tdim = tdim;

  // Set cell type
  mesh._cell_type = CellType::create(type);

  // Initialize topological dimension
  mesh._topology.init(tdim);

  // Initialize domains
  mesh._domains.init(tdim);

  // Initialize temporary storage for local cell data
  vertices = std::vector<std::size_t>(mesh.type().num_vertices(tdim), 0);
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, std::string type, std::size_t tdim, std::size_t gdim)
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
    dolfin_error("MeshEditor.cpp",
                 "open mesh for editing",
                 "Unknown cell type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void MeshEditor::init_vertices(std::size_t num_vertices)
{
  // Check if we are currently editing a mesh
  if (!mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "initialize vertices in mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Initialize mesh data
  this->num_vertices = num_vertices;
  mesh->_topology.init(0,    num_vertices);
  mesh->_topology.init_global_indices(0, num_vertices);
  mesh->_geometry.init(gdim, num_vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::init_cells(std::size_t num_cells)
{
  // Check if we are currently editing a mesh
  if (!mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "initialize cells in mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Initialize mesh data
  this->num_cells = num_cells;
  mesh->_topology.init(tdim, num_cells);
  mesh->_topology.init_global_indices(tdim, num_cells);
  mesh->_topology(tdim, 0).init(num_cells, mesh->type().num_vertices(tdim));
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
  dolfin_assert(gdim == 1);
  std::vector<double> p(1);
  p[0] = x;
  add_vertex(index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, double x, double y)
{
  dolfin_assert(gdim == 2);
  std::vector<double> p(2);
  p[0] = x;
  p[1] = y;
  add_vertex(index, p);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex(std::size_t index, double x, double y, double z)
{
  dolfin_assert(gdim == 3);
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
  // Geometric dimension
  const std::size_t gdim = mesh->geometry().dim();

  // Add vertex
  add_vertex_common(local_index, gdim);

  // Set coordinate
  std::vector<double> x(p.coordinates(), p.coordinates() + gdim);
  mesh->_geometry.set(local_index, x);
  mesh->_topology.set_global_index(0, local_index, global_index);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex_global(std::size_t local_index,
                                   std::size_t global_index,
                                   const std::vector<double>& x)
{
  // Add vertex
  add_vertex_common(local_index, x.size());

  // Set coordinate
  mesh->_geometry.set(local_index, x);
  mesh->_topology.set_global_index(0, local_index, global_index);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1)
{
  dolfin_assert(tdim == 1);
  std::vector<std::size_t> vertices(2);
  vertices[0] = v0;
  vertices[1] = v1;
  add_cell(c, c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                          std::size_t v2)
{
  dolfin_assert(tdim == 2);
  std::vector<std::size_t> vertices(3);
  vertices[0] = v0;
  vertices[1] = v1;
  vertices[2] = v2;
  add_cell(c, c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, std::size_t v0, std::size_t v1,
                          std::size_t v2, std::size_t v3)
{
  dolfin_assert(tdim == 3);
  std::vector<std::size_t> vertices(4);
  vertices[0] = v0;
  vertices[1] = v1;
  vertices[2] = v2;
  vertices[3] = v3;
  add_cell(c, c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t c, const std::vector<std::size_t>& v)
{
  add_cell(c, c, v);
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell(std::size_t local_index, std::size_t global_index,
                          const std::vector<std::size_t>& v)
{
  dolfin_assert(v.size() == tdim + 1);

  // Check vertices
  check_vertices(v);

  // Add cell
  add_cell_common(local_index, tdim);

  // Set data
  mesh->_topology(tdim, 0).set(local_index, v);
  mesh->_topology.set_global_index(tdim, local_index, global_index);
}
//-----------------------------------------------------------------------------
void MeshEditor::close(bool order)
{
  // Order mesh if requested
  dolfin_assert(mesh);
  if (order && !mesh->ordered())
    mesh->order();

  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void MeshEditor::add_vertex_common(std::size_t v, std::size_t gdim)
{
  // Check if we are currently editing a mesh
  if (!mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Check that the dimension matches
  if (gdim != this->gdim)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Illegal dimension for vertex coordinate (%d), expecting %d",
                 gdim, this->gdim);
  }

  // Check value of vertex index
  if (v >= num_vertices)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Vertex index (%d) out of range [0, %d)",
                 v, num_vertices);
  }

  // Check if there is room for more vertices
  if (next_vertex >= num_vertices)
  {
    dolfin_error("MeshEditor.cpp",
                 "add vertex to mesh using mesh editor",
                 "Vertex list is full, %d vertices already specified",
                 num_vertices);
  }

  // Step to next vertex
  next_vertex++;
}
//-----------------------------------------------------------------------------
void MeshEditor::add_cell_common(std::size_t c, std::size_t tdim)
{
  // Check if we are currently editing a mesh
  if (!mesh)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "No mesh opened, unable to edit");
  }

  // Check that the dimension matches
  if (tdim != this->tdim)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Illegal dimension for cell (%d), expecting %d", tdim, this->tdim);
  }

  // Check value of cell index
  if (c >= num_cells)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Cell index (%d) out of range [0, %d)",
                 c, num_cells);
  }

  // Check if there is room for more cells
  if (next_cell >= num_cells)
  {
    dolfin_error("MeshEditor.cpp",
                 "add cell to mesh using mesh editor",
                 "Cell list is full, %d cells already specified",
                 num_cells);
  }

  // Step to next cell
  next_cell++;
}
//-----------------------------------------------------------------------------
void MeshEditor::clear()
{
  tdim = 0;
  gdim = 0;
  num_vertices = 0;
  num_cells = 0;
  next_vertex = 0;
  next_cell = 0;
  mesh = 0;
  vertices.clear();
}
//-----------------------------------------------------------------------------
void MeshEditor::check_vertices(const std::vector<std::size_t>& v) const
{
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    if (num_vertices > 0 && v[i] >= num_vertices)
    {
      dolfin_error("MeshEditor.cpp",
                   "add cell using mesh editor",
                   "Vertex index (%d) out of range [0, %d)", v[i], num_vertices);
    }
  }
}
//-----------------------------------------------------------------------------
