// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-15
//
// Modified by Anders Logg, 2008.

#include <tr1/memory>
#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/CellType.h>
#include "XMLLocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLLocalMeshData::XMLLocalMeshData(LocalMeshData& mesh_data)
  : XMLObject(), mesh_data(mesh_data), state(OUTSIDE)
{
  // Get number of processes and process number
  mesh_data.num_processes = MPI::num_processes();
  mesh_data.process_number = MPI::process_number();
}
//-----------------------------------------------------------------------------
XMLLocalMeshData::~XMLLocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch (state)
  {
  case OUTSIDE:

    if (xmlStrcasecmp(name, (xmlChar* ) "dolfin") == 0)
    {
      state = INSIDE_DOLFIN;
    }

    break;

  case INSIDE_DOLFIN:

    if (xmlStrcasecmp(name, (xmlChar* ) "mesh") == 0)
    {
      readMesh(name, attrs);
      state = INSIDE_MESH;
    }
    
    break;

  case INSIDE_MESH:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "vertices") == 0)
    {
      readVertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "cells") == 0)
    {
      readCells(name, attrs);
      state = INSIDE_CELLS;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "data") == 0)
    {
      error("Unable to read auxiliary mesh data in parallel, not implemented (yet).");
      state = INSIDE_DATA;
    }

    break;
    
  case INSIDE_VERTICES:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "vertex") == 0)
      readVertex(name, attrs);

    break;
    
  case INSIDE_CELLS:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "interval") == 0)
      readInterval(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar* ) "triangle") == 0)
      readTriangle(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar* ) "tetrahedron") == 0)
      readTetrahedron(name, attrs);
    
    break;

  case INSIDE_DATA:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "meshfunction") == 0)
    {
      readMeshFunction(name, attrs);
      state = INSIDE_MESH_FUNCTION;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "array") == 0)
    {
      readArray(name, attrs);
      state = INSIDE_ARRAY;
    }

    break;

  case INSIDE_MESH_FUNCTION:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "entity") == 0)
      readMeshEntity(name, attrs);

    break;

  case INSIDE_ARRAY:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "element") == 0)
      readArrayElement(name, attrs);

    break;

  default:
    error("Inconsistent state in XML reader: %d.", state);
  }
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::endElement(const xmlChar* name)
{
  switch (state)
  {
  case INSIDE_DOLFIN:

    if (xmlStrcasecmp(name, (xmlChar* ) "dolfin") == 0)
    {
      state = DONE;
    }

    break;

  case INSIDE_MESH:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "mesh") == 0)
    {
      state = INSIDE_DOLFIN;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if (xmlStrcasecmp(name, (xmlChar* ) "vertices") == 0)
    {
      state = INSIDE_MESH;    
    }

    break;

  case INSIDE_CELLS:
	 
    if (xmlStrcasecmp(name, (xmlChar* ) "cells") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_DATA:

    if (xmlStrcasecmp(name, (xmlChar* ) "data") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_MESH_FUNCTION:

    if (xmlStrcasecmp(name, (xmlChar* ) "meshfunction") == 0)
    {
      state = INSIDE_DATA;
    }

    break;

  case INSIDE_ARRAY:

    if (xmlStrcasecmp(name, (xmlChar* ) "array") == 0)
    {
      state = INSIDE_DATA;
    }

    break;

  default:
    error("Closing XML tag '%s', but state is %d.", name, state);
  }

}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool XMLLocalMeshData::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readMesh(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  std::string type = parseString(name, attrs, "celltype");
  gdim = parseUnsignedInt(name, attrs, "dim");
  
  // Create cell type to get topological dimension
  std::auto_ptr<CellType> cell_type(CellType::create(type));
  
  tdim = cell_type->dim();

  // Get number of entities for topological dimension 0
  //num_cell_vertices = cell_type->numEntities(0);
  mesh_data.cell_type = CellType::create(type);
  mesh_data.tdim = tdim;
  mesh_data.gdim = gdim;

  // Clear all data
  mesh_data.clear();
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readVertices(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global vertices
  const uint num_global_vertices = parseUnsignedInt(name, attrs, "size");
  dolfin_debug1("num_global_vertices = %d", num_global_vertices);
  mesh_data.num_global_vertices = num_global_vertices;

  // Compute vertex range
  mesh_data.initial_vertex_range(vertex_index_start, vertex_index_stop);

  dolfin_debug3("vertex range: [%d, %d] size = %d",
                vertex_index_start, vertex_index_stop, num_local_vertices());

  // Reserve space for local-to-global vertex map and vertex coordinates
  mesh_data.vertex_indices.reserve(num_local_vertices());
  mesh_data.vertex_coordinates.reserve(num_local_vertices());
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readVertex(const xmlChar* name, const xmlChar** attrs)
{
  // Read vertex index
  const uint v = parseUnsignedInt(name, attrs, "index");

  // Skip vertices not in range for this process
  if (v < vertex_index_start || v > vertex_index_stop)
    return;

  std::vector<double> coordinate;
  
  // Parse vertex coordinates
  switch (gdim)
  {
  case 1:
    {
      coordinate.push_back(parseReal(name, attrs, "x"));
      mesh_data.vertex_coordinates.push_back(coordinate);
    }
  break;
  case 2:
    {
      coordinate.reserve(2);
      coordinate.push_back(parseReal(name, attrs, "x"));
      coordinate.push_back(parseReal(name, attrs, "y"));
      mesh_data.vertex_coordinates.push_back(coordinate);
    }
    break;
  case 3:
    {
      coordinate.reserve(3);
      coordinate.push_back(parseReal(name, attrs, "x"));
      coordinate.push_back(parseReal(name, attrs, "y"));
      coordinate.push_back(parseReal(name, attrs, "z"));
      mesh_data.vertex_coordinates.push_back(coordinate);
    }
    break;
  default:
    error("Geometric dimension of mesh must be 1, 2 or 3.");
  }

  // Store global vertex numbering 
  mesh_data.vertex_indices.push_back(v);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readCells(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global cells 
  const uint num_global_cells = parseUnsignedInt(name, attrs, "size");
  dolfin_debug1("num_global_cells = %d", num_global_cells);
  mesh_data.num_global_cells = num_global_cells;

  // Compute cell range
  mesh_data.initial_cell_range(cell_index_start, cell_index_stop);

  dolfin_debug3("cell range: [%d, %d] size = %d",
                cell_index_start, cell_index_stop, num_local_cells());

  // Reserve space for cells
  mesh_data.cell_vertices.reserve(num_local_cells());
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readInterval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parseUnsignedInt(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_index_start || c > cell_index_stop)
    return;

  // Parse values
  std::vector<uint> cell(2);
  cell[0] = parseUnsignedInt(name, attrs, "v0");
  cell[1] = parseUnsignedInt(name, attrs, "v1");

  // Add cell
  mesh_data.cell_vertices.push_back(cell);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 2)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parseUnsignedInt(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_index_start || c > cell_index_stop)
    return;

  // Parse values
  std::vector<uint> cell(3);
  cell[0] = parseUnsignedInt(name, attrs, "v0");
  cell[1] = parseUnsignedInt(name, attrs, "v1");
  cell[2] = parseUnsignedInt(name, attrs, "v2");
  
  // Add cell
  mesh_data.cell_vertices.push_back(cell);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 3)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parseUnsignedInt(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_index_start || c > cell_index_stop)
    return;

  // Parse values
  std::vector<uint> cell(4);
  cell[0] = parseUnsignedInt(name, attrs, "v0");
  cell[1] = parseUnsignedInt(name, attrs, "v1");
  cell[2] = parseUnsignedInt(name, attrs, "v2");
  cell[3] = parseUnsignedInt(name, attrs, "v3");
  
  // Add cell
  mesh_data.cell_vertices.push_back(cell);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readMeshFunction(const xmlChar* name, const xmlChar** attrs)
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readArray(const xmlChar* name, const xmlChar** attrs)
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readMeshEntity(const xmlChar* name, const xmlChar** attrs)
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readArrayElement(const xmlChar* name, const xmlChar** attrs)
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::num_local_vertices() const
{
  return vertex_index_stop - vertex_index_start + 1;
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::num_local_cells() const
{
  return cell_index_stop - cell_index_start + 1;
}
//-----------------------------------------------------------------------------
