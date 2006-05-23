// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-16
// Last changed: 2006-05-18

#include <dolfin/dolfin_log.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshEditor.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEditor::MeshEditor()
  : dim(0),
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
void MeshEditor::edit(NewMesh& mesh, uint dim)
{
  // Clear old mesh data
  mesh.data.clear();
  clear();

  // Save mesh and dimension
  this->mesh = &mesh;
  this->dim = dim;

  // Initialize topological dimension
  mesh.data.topology.init(dim);
}
//-----------------------------------------------------------------------------
void MeshEditor::initVertices(uint num_vertices)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");
  
  // Initialize mesh data
  this->num_vertices = num_vertices;
  mesh->data.topology.init(0,   num_vertices);
  mesh->data.geometry.init(dim, num_vertices);
}
//-----------------------------------------------------------------------------
void MeshEditor::initCells(uint num_cells)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Initialize mesh data
  this->num_cells = num_cells;
  mesh->data.topology.init(dim, num_cells);

  // Initialize temporary storage with illegal values, check later
  cells.reserve(num_cells);
  Array<uint> cell(dim + 1);
  for (uint i = 0; i < num_cells; i++)
  {
    for (uint j = 0; j < dim + 1; j++)
      cell[j] = num_vertices;
    cells.push_back(cell);
  }
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
void MeshEditor::addCell(uint c, uint v0, uint v1)
{
  // Add cell
  addCellCommon(c, 1);

  // Set data
  cells[c][0] = v0;
  cells[c][1] = v1;
}
//--------------------------------------------------------------------------
void MeshEditor::addCell(uint c, uint v0, uint v1, uint v2)
{
  // Add cell
  addCellCommon(c, 2);

  // Set data
  cells[c][0] = v0;
  cells[c][1] = v1;
  cells[c][2] = v2;
}
//-----------------------------------------------------------------------------
void MeshEditor::addCell(uint c, uint v0, uint v1, uint v2, uint v3)
{
  // Add cell
  addCellCommon(c, 3);

  // Set data
  cells[c][0] = v0;
  cells[c][1] = v1;
  cells[c][2] = v2;
  cells[c][3] = v3;
}
//-----------------------------------------------------------------------------
void MeshEditor::close()
{
  // FIXME: Init mesh?

  // Check that everything was added
  for (uint c = 0; c < num_cells; c++)
    for (uint v = 0; v < dim + 1; v++)
      if ( cells[c][v] >= num_vertices )
	dolfin_error("Cell connectivity out of range or not completely specified.");
  
  // Set connectivity
  mesh->data.topology.set(dim, 0, cells);

  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void MeshEditor::addVertexCommon(uint v, uint dim)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Check that the dimension matches
  if ( dim != this->dim )
    dolfin_error2("Illegal dimension for vertex coordinate: %d (should be %d).",
		  dim, this->dim);

  // Check value of vertex index
  if ( v >= num_vertices )
    dolfin_error2("Vertex index (%d) out of range [0, %d].",
		  v, num_vertices - 1);

  // Check if there is room for more vertices
  if ( next_vertex >= num_vertices )
    dolfin_error1("Vertex list is full, %d vertices already specified.",
		  num_vertices);
  
  // Store numbering
  // FIXME: Don't know how to do this

  // Step to next vertex
  next_vertex++;
}
//-----------------------------------------------------------------------------
void MeshEditor::addCellCommon(uint c, uint dim)
{
  // Check if we are currently editing a mesh
  if ( !mesh )
    dolfin_error("No mesh opened, unable to edit.");

  // Check that the dimension matches
  if ( dim != this->dim )
    dolfin_error2("Illegal dimension for cell: %d (should be %d).",
		  dim, this->dim);

  // Check value of vertex index
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
  dim = 0;
  num_vertices = 0;
  num_cells = 0;
  next_vertex = 0;
  next_cell = 0;
  mesh = 0;

  for (uint i = 0; i < cells.size(); i++)
    cells[i].clear();
  cells.clear();
}
//-----------------------------------------------------------------------------
