// Copyright (C) 2008 Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-11-28
// Last changed: 2011-03-30
//
// Modified by Anders Logg, 2008.
// Modified by Kent-Andre Mardal, 2011.

#include <boost/assign/list_of.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include "XMLLocalMeshData.h"
#include "XMLArray.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLLocalMeshData::XMLLocalMeshData(LocalMeshData& mesh_data, XMLFile& parser)
  : XMLHandler(parser), mesh_data(mesh_data), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLLocalMeshData::~XMLLocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch (state)
  {
  case OUTSIDE:

    if (xmlStrcasecmp(name, (xmlChar* ) "mesh") == 0)
    {
      read_mesh(name, attrs);
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_MESH:

    if (xmlStrcasecmp(name, (xmlChar* ) "vertices") == 0)
    {
      read_vertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "cells") == 0)
    {
      read_cells(name, attrs);
      state = INSIDE_CELLS;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "data") == 0)
    {
      state = INSIDE_DATA;
      read_mesh_data(name, attrs);
    }

    break;

  case INSIDE_VERTICES:

    if (xmlStrcasecmp(name, (xmlChar* ) "vertex") == 0)
      read_vertex(name, attrs);

    break;

  case INSIDE_CELLS:

    if (xmlStrcasecmp(name, (xmlChar* ) "interval") == 0)
      read_interval(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar* ) "triangle") == 0)
      read_triangle(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar* ) "tetrahedron") == 0)
      read_tetrahedron(name, attrs);

    break;

  case INSIDE_DATA:

    if (xmlStrcasecmp(name, (xmlChar* ) "meshfunction") == 0)
    {
      read_mesh_function(name, attrs);
      state = INSIDE_MESH_FUNCTION;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "array") == 0)
    {
      read_array(name, attrs);
      state = INSIDE_ARRAY;
    }
    else if (xmlStrcasecmp(name, (xmlChar* ) "data_entry") == 0) 
    {
      read_data_entry(name, attrs); 
      state = INSIDE_DATA_ENTRY; 
    }

    break; 

  case INSIDE_DATA_ENTRY: 
    if (xmlStrcasecmp(name, (xmlChar* ) "array") == 0) 
    {
      read_array(name, attrs); 
      state = INSIDE_ARRAY; 
    }

    break; 

  default:
    error("Inconsistent state in XML reader: %d.", state);
  }
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::end_element(const xmlChar* name)
{
  switch (state)
  {

  case INSIDE_MESH:

    if (xmlStrcasecmp(name, (xmlChar* ) "mesh") == 0)
    {
      state = DONE;
      release();
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
  
  case INSIDE_DATA_ENTRY: 

    if (xmlStrcasecmp(name, (xmlChar* ) "data_entry") == 0)
    {
      state = INSIDE_DATA;
    }


  case INSIDE_ARRAY:

    if (xmlStrcasecmp(name, (xmlChar* ) "array") == 0)
    {
      state = INSIDE_DATA_ENTRY;
    }

    if (xmlStrcasecmp(name, (xmlChar* ) "data_entry") == 0)
    {
      state = INSIDE_DATA;
    }


    break;

  default:
    error("Closing XML tag '%s', but state is %d.", name, state);
  }

}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_mesh(const xmlChar* name, const xmlChar** attrs)
{
  // Clear all data
  mesh_data.clear();

  // Parse values
  std::string type = parse_string(name, attrs, "celltype");
  gdim = parse_uint(name, attrs, "dim");

  // Create cell type to get topological dimension
  boost::scoped_ptr<CellType> cell_type(CellType::create(type));
  tdim = cell_type->dim();

  // Get number of entities for topological dimension 0
  mesh_data.tdim = tdim;
  mesh_data.gdim = gdim;
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_vertices(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global vertices
  const uint num_global_vertices = parse_uint(name, attrs, "size");
  mesh_data.num_global_vertices = num_global_vertices;

  // Compute vertex range
  vertex_range = MPI::local_range(num_global_vertices);

  // Reserve space for local-to-global vertex map and vertex coordinates
  mesh_data.vertex_indices.reserve(num_local_vertices());
  mesh_data.vertex_coordinates.reserve(num_local_vertices());
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_vertex(const xmlChar* name, const xmlChar** attrs)
{
  // Read vertex index
  const uint v = parse_uint(name, attrs, "index");

  // Skip vertices not in range for this process
  if (v < vertex_range.first || v >= vertex_range.second)
    return;

  // Parse vertex coordinates
  switch (gdim)
  {
  case 1:
    {
      const std::vector<double> coordinate = boost::assign::list_of(parse_float(name, attrs, "x"));
      mesh_data.vertex_coordinates.push_back(coordinate);
    }
  break;
  case 2:
    {
      const std::vector<double> coordinate = boost::assign::list_of(parse_float(name, attrs, "x"))
                                                                   (parse_float(name, attrs, "y"));
      mesh_data.vertex_coordinates.push_back(coordinate);
    }
    break;
  case 3:
    {
      const std::vector<double> coordinate = boost::assign::list_of(parse_float(name, attrs, "x"))
                                                                   (parse_float(name, attrs, "y"))
                                                                   (parse_float(name, attrs, "z"));
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
void XMLLocalMeshData::read_cells(const xmlChar* name, const xmlChar** attrs)
{
  // Parse the number of global cells
  const uint num_global_cells = parse_uint(name, attrs, "size");
  mesh_data.num_global_cells = num_global_cells;

  // Compute cell range
  cell_range = MPI::local_range(num_global_cells);

  // Reserve space for cells
  mesh_data.cell_vertices.reserve(num_local_cells());

  // Reserve space for global cell indices
  mesh_data.global_cell_indices.reserve(num_local_cells());
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_interval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parse_uint(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_range.first || c >= cell_range.second)
    return;

  // Parse values
  std::vector<uint> cell(2);
  cell[0] = parse_uint(name, attrs, "v0");
  cell[1] = parse_uint(name, attrs, "v1");

  // Add cell
  mesh_data.cell_vertices.push_back(cell);

  // Add global cell index
  mesh_data.global_cell_indices.push_back(c);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_triangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 2)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parse_uint(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_range.first || c >= cell_range.second)
    return;

  // Parse values
  std::vector<uint> cell(3);
  cell[0] = parse_uint(name, attrs, "v0");
  cell[1] = parse_uint(name, attrs, "v1");
  cell[2] = parse_uint(name, attrs, "v2");

  // Add cell
  mesh_data.cell_vertices.push_back(cell);

  // Add global cell index
  mesh_data.global_cell_indices.push_back(c);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_tetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 3)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Read cell index
  const uint c = parse_uint(name, attrs, "index");

  // Skip cells not in range for this process
  if (c < cell_range.first || c >= cell_range.second)
    return;

  // Parse values
  std::vector<uint> cell(4);
  cell[0] = parse_uint(name, attrs, "v0");
  cell[1] = parse_uint(name, attrs, "v1");
  cell[2] = parse_uint(name, attrs, "v2");
  cell[3] = parse_uint(name, attrs, "v3");

  // Add cell
  mesh_data.cell_vertices.push_back(cell);

  // Add global cell index
  mesh_data.global_cell_indices.push_back(c);
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_mesh_function(const xmlChar* name, const xmlChar** attrs)
{
  error("Local mesh data can not read mesh functions.");
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_array(const xmlChar* name, const xmlChar** attrs)
{
  std::string array_type = parse_string(name, attrs, "type");
  uint size = parse_uint(name, attrs, "size");

  XMLArray* xml_array = 0;

  if ( array_type.compare("int") == 0 )
  {
    // FIXME: Add support for more types in MeshData?
    std::vector<int>* ux = new std::vector<int>();
    if ( xml_array ) {
      delete xml_array;
      xml_array = 0; 
    }
    xml_array = new XMLArray(*ux, parser, size);
    xml_array->handle();
  }
  else if ( array_type.compare("uint") == 0 )
  {
    //FIXME this is uint 
    std::vector<unsigned int>* ux = new std::vector<unsigned int>();
    if ( xml_array ) {
      delete xml_array;
      xml_array = 0; 
    }
    xml_array = new XMLArray(*ux, parser, size);
    xml_array->handle();
    mesh_data.arrays[data_entry_name] = ux; 
  }
  else if ( array_type.compare("double") == 0 )
  {
    // FIXME: Add support for more types in MeshData?
    std::vector<double>* dx = new std::vector<double>();
    if ( xml_array ) {
      delete xml_array;
      xml_array = 0; 
    }
    xml_array = new XMLArray(*dx, parser, size);
    xml_array->handle();
  }
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_mesh_data(const xmlChar* name, const xmlChar** attrs)
{
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::read_data_entry(const xmlChar* name, const xmlChar** attrs)
{
  data_entry_name = parse_string(name, attrs, "name");
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::num_local_vertices() const
{
  return vertex_range.second - vertex_range.first;
}
//-----------------------------------------------------------------------------
dolfin::uint XMLLocalMeshData::num_local_cells() const
{
  return cell_range.second - cell_range.first;
}
//-----------------------------------------------------------------------------
