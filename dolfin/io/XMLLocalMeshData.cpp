// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-02
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
  // Do nothing
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

  // Clear all data
  mesh_data.clear();
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readVertices(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global vertices
  const uint num_global_vertices = parseUnsignedInt(name, attrs, "size");

  // Get process number and number of processes
  const uint num_processes  = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Compute number of vertices per process and remainder
  const uint n = num_global_vertices / num_processes;
  const uint r = num_global_vertices % num_processes;

  if (process_number < r)
  {
    vertex_index_start = process_number*(n + 1);
    vertex_index_stop = vertex_index_start + n;
  }
  else
  {
    vertex_index_start = process_number*n + r;
    vertex_index_stop = vertex_index_start + n - 1;
  }

  // Reserve space for local-to-global vertex map and vertex coordinates
  mesh_data.vertex_indices.reserve(num_local_vertices());
  mesh_data.vertex_coordinates.reserve(gdim * num_local_vertices());

  dolfin_debug2("Reading %d vertices out of %d vertices.",
                num_local_vertices(), num_global_vertices);
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
  dolfin::cout << "num_global_cells = " << num_global_cells << dolfin::endl;

  // Get process number and number of processes
  const uint num_processes  = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Compute number of cells per process and remainder
  const uint n = num_global_cells / num_processes;
  const uint r = num_global_cells % num_processes;

  if (process_number < r)
  {
    cell_index_start = process_number*(n + 1);
    cell_index_stop = cell_index_start + n;
  }
  else
  {
    cell_index_start = process_number*n + r;
    cell_index_stop = cell_index_start + n - 1;
  }

  dolfin::cout << "cell_index_start = " << cell_index_start << dolfin::endl;
  dolfin::cout << "cell_index_stop = " << cell_index_stop << dolfin::endl;
  dolfin::cout << "num_local_cells  = " << num_local_cells() << dolfin::endl;

  // Reserve space for cells
  mesh_data.cell_vertices.reserve(num_local_cells());

  dolfin_debug2("Reading %d out of %d cells.",
                num_local_cells(), num_global_cells);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readInterval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Parse values
  parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  
  std::vector<uint> cell(2);
  cell.push_back(v0);
  cell.push_back(v1);

  // Add cell
  mesh_data.cell_vertices.push_back(cell);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 2)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Parse values
  parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  
  std::vector<uint> cell(3);
  cell.push_back(v0);
  cell.push_back(v1);
  cell.push_back(v2);

  // Add cell
  mesh_data.cell_vertices.push_back(cell);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 3)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Parse values
  parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  uint v3 = parseUnsignedInt(name, attrs, "v3");
  
  std::vector<uint> cell(4);
  cell.push_back(v0);
  cell.push_back(v1);
  cell.push_back(v2);
  cell.push_back(v3);

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
