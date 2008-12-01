// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-11-28

#include "XMLLocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLLocalMeshData::XMLLocalMeshData(LocalMeshData& mesh_data)
  : mesh_data(mesh_data)
{
  // Do nothing
}

//-----------------------------------------------------------------------------
XMLLocalMeshData::~XMLLocalMeshData()
{
  // Do nothing
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
  num_cell_vertices = cell_type->numEntities(0);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::readVertices(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global vertices
  const uint num_global_vertices = parseUnsignedInt(name, attrs, "size");

  // Get process number// and number of processes
  const uint num_processes = MPI::num_processes();
  //const uint process_number = MPI::process_number();

  // Compute number of vertices per process and remainder
  const uint n = num_global_vertices / num_processes;
  const uint r = num_global_vertices % num_processes;

  // Distribute remainder evenly among first r processes
  vertex_distribution = new uint[num_processes + 1];
  uint offset = 0;
  for (uint p = 0; p < num_processes + 1; p++)
  {
    vertex_distribution[p] = offset;
    if (p < r)
      offset += n + 1;
    else
      offset += n;
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
  if (v < first_local_vertex() || v > last_local_vertex())
    return;
  
  // Parse vertex coordinates
  switch (gdim)
  {
  case 1:
    {
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "x"));
    }
  break;
  case 2:
    {
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "x"));
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "y"));
    }
    break;
  case 3:
    {
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "x"));
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "y"));
      mesh_data.vertex_coordinates.push_back(parseReal(name, attrs, "z"));
    }
    break;
  default:
    error("Geometric dimension of mesh must be 1, 2 or 3.");
  }

  // Store global vertex numbering 
  mesh_data.vertex_indices.push_back(v);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::closeVertices()
{
  /* Do nothing */
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::num_local_vertices() const
{
  dolfin_assert(vertex_distribution);
  const uint process_number = MPI::process_number();
  return vertex_distribution[process_number + 1] - vertex_distribution[process_number];
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::first_local_vertex() const
{
  dolfin_assert(vertex_distribution);
  const uint process_number = MPI::process_number();
  return vertex_distribution[process_number];
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::last_local_vertex() const
{
  dolfin_assert(vertex_distribution);
  const uint process_number = MPI::process_number();
  return vertex_distribution[process_number + 1] - 1;
}

//-----------------------------------------------------------------------------
void XMLLocalMeshData::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch (state)
  {
  case OUTSIDE:

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
