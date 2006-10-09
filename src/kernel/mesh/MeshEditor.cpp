// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-16
// Last changed: 2006-06-22

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Point.h>
#include <dolfin/MeshEditor.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEditor::MeshEditor()
  : tdim(0), gdim(0),
    num_vertices(0), num_cells(0),
    next_vertex(0), next_cell(0),
    mesh(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEditor::~MeshEditor()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshEditor::open(Mesh& mesh, CellType::Type type, uint tdim, uint gdim)
{
  // Clear old mesh data
  mesh.data.clear();
  clear();

  // Save mesh and dimension
  this->mesh = &mesh;
  this->gdim = gdim;
  this->tdim = tdim;

  // Set cell type
  mesh.data.cell_type = CellType::create(type);

  // Initialize topological dimension
  mesh.data.topology.init(tdim);

  // Initialize temporary storage for local cell data
  vertices.reserve(mesh.type().numVertices(tdim));
  for (uint i = 0; i < mesh.type().numVertices(tdim); i++)
    vertices.push_back(0);
}
//-----------------------------------------------------------------------------
void MeshEditor::initVertices(uint num_vertices)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");
  
  // Initialize mesh data
  this->num_vertices = num_vertices;
  mesh->data.topology.init(0,    num_vertices);
  mesh->data.geometry.init(gdim, num_vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::initCells(uint num_cells)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Initialize mesh data
  this->num_cells = num_cells;
  mesh->data.topology.init(tdim, num_cells);
  mesh->data.topology(tdim, 0).init(num_cells, mesh->type().numVertices(tdim));
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertex(uint v, const Point& p)
{
  // Add vertex
  addVertexCommon(v, mesh->geometry().dim());
  
  // Set coordinate
  for (uint i = 0; i < mesh->geometry().dim(); i++)
    mesh->data.geometry.set(v, i, p[i]);
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertex(uint v, real x)
{
  // Add vertex
  addVertexCommon(v, 1);

  // Set coordinate
  mesh->data.geometry.set(next_vertex, 0, x);
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertex(uint v, real x, real y)
{
  // Add vertex
  addVertexCommon(v, 2);

  // Set coordinate
  mesh->data.geometry.set(v, 0, x);
  mesh->data.geometry.set(v, 1, y);
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertex(uint v, real x, real y, real z)
{
  // Add vertex
  addVertexCommon(v, 3);

  // Set coordinate
  mesh->data.geometry.set(v, 0, x);
  mesh->data.geometry.set(v, 1, y);
  mesh->data.geometry.set(v, 2, z);
}
//-----------------------------------------------------------------------------
void MeshEditor::addCell(uint c, const Array<uint>& v)
{
  // Add cell
  addCellCommon(c, tdim);

  // Set data
  mesh->data.topology(tdim, 0).set(c, v);
}
//-----------------------------------------------------------------------------
void MeshEditor::addCell(uint c, uint v0, uint v1)
{
  // Add cell
  addCellCommon(c, 1);

  // Set data
  vertices[0] = v0;
  vertices[1] = v1;
  mesh->data.topology(tdim, 0).set(c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::addCell(uint c, uint v0, uint v1, uint v2)
{
  // Add cell
  addCellCommon(c, 2);

  // Set data
  vertices[0] = v0;
  vertices[1] = v1;
  vertices[2] = v2;
  mesh->data.topology(tdim, 0).set(c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::addCell(uint c, uint v0, uint v1, uint v2, uint v3)
{
  // Add cell
  addCellCommon(c, 3);

  // Set data
  vertices[0] = v0;
  vertices[1] = v1;
  vertices[2] = v2;
  vertices[3] = v3;
  mesh->data.topology(tdim, 0).set(c, vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::close()
{
  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertexCommon(uint v, uint gdim)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Check that the dimension matches
  if ( gdim != this->gdim )
    dolfin_error2("Illegal dimension for vertex coordinate: %d (should be %d).",
		  gdim, this->gdim);

  // Check value of vertex index
  if ( v >= num_vertices )
    dolfin_error2("Vertex index (%d) out of range [0, %d].",
		  v, num_vertices - 1);

  // Check if there is room for more vertices
  if ( next_vertex >= num_vertices )
    dolfin_error1("Vertex list is full, %d vertices already specified.",
		  num_vertices);
  
  // Step to next vertex
  next_vertex++;
}
//-----------------------------------------------------------------------------
void MeshEditor::addCellCommon(uint c, uint tdim)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Check that the dimension matches
  if ( tdim != this->tdim )
    dolfin_error2("Illegal dimension for cell: %d (should be %d).",
		  tdim, this->tdim);

  // Check value of cell index
  if ( c >= num_cells )
    dolfin_error2("Cell index (%d) out of range [0, %d].",
		  c, num_cells - 1);

  // Check if there is room for more cells
  if ( next_cell >= num_cells )
    dolfin_error1("Cell list is full, %d cells already specified.", num_cells);

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
