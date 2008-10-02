// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-22
// Last changed: 2008-09-22

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
  if ( type == "point" )
    open(mesh, CellType::point, tdim, gdim);
  else if ( type == "interval" )
    open(mesh, CellType::interval, tdim, gdim);
  else if ( type == "triangle" )
    open(mesh, CellType::triangle, tdim, gdim);
  else if ( type == "tetrahedron" )
    open(mesh, CellType::tetrahedron, tdim, gdim);
  else
    error("Unknown cell type \"%s\".", type.c_str());
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addVertex(uint v, const Point& p)
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
void DynamicMeshEditor::addVertex(uint v, double x)
{
  Point p(x);
  addVertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addVertex(uint v, double x, double y)
{
  Point p(x, y);
  addVertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addVertex(uint v, double x, double y, double z)
{
  Point p(x, y, z);
  addVertex(v, p);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addCell(uint c, const Array<uint>& v)
{
  // Check size of array
  const uint vertices_per_cell = cell_type->numVertices(tdim);
  if (v.size() != vertices_per_cell)
  {
    error("Illegal number of vertices (%d) for cell, expected %d.",
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
void DynamicMeshEditor::addCell(uint c, uint v0, uint v1)
{
  Array<uint> vertices(v0, v1);
  addCell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addCell(uint c, uint v0, uint v1, uint v2)
{
  Array<uint> vertices(v0, v1, v2);
  addCell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::addCell(uint c, uint v0, uint v1, uint v2, uint v3)
{
  Array<uint> vertices(v0, v1, v2, v3);
  addCell(c, vertices);
}
//-----------------------------------------------------------------------------
void DynamicMeshEditor::close()
{
  dolfin_assert(mesh);
  dolfin_assert(cell_type);

  // Open default mesh editor
  MeshEditor editor;
  editor.open(*mesh, cell_type->cellType(), tdim, gdim);

  // Set number of vertices
  const uint num_vertices = vertex_coordinates.size() / gdim;
  editor.initVertices(num_vertices);

  // Set number of cells
  const uint vertices_per_cell = cell_type->numVertices(gdim);
  const uint num_cells = cell_vertices.size() / vertices_per_cell;
  editor.initCells(num_cells);

  // Add vertices
  Point p;
  for (uint v = 0; v < num_vertices; v++)
  {
    const uint offset = v*gdim;
    for (uint i = 0; i < gdim; i++)
      p[i] = vertex_coordinates[offset + i];
    editor.addVertex(v, p);
  }

  // Add cells
  Array<uint> vertices(vertices_per_cell);
  for (uint c = 0; c < num_cells; c++)
  {
    const uint offset = c*vertices_per_cell;
    for (uint i = 0; i < vertices_per_cell; i++)
      vertices[i] = cell_vertices[offset + i];
    editor.addCell(c, vertices);
  }

  // Close editor
  editor.close();

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
