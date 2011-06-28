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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-11-28
// Last changed: 2011-05-30
//
// Modified by Anders Logg, 2008.
// Modified by Kent-Andre Mardal, 2011.

#include <boost/assign/list_of.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include "XMLLocalMeshDataDistributed.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLLocalMeshDataDistributed::read(LocalMeshData& mesh_data,
                                       const std::string filename)
{
  std::cout << "Inside new read function" << std::endl;

  // Create handler
  xmlSAXHandler sax_handler;
  sax_handler.initialized = XML_SAX2_MAGIC;

  sax_handler.startDocument = sax_start_document;
  sax_handler.endDocument   = sax_end_document;

  sax_handler.startElement  = sax_start_element;
  sax_handler.endElement    = sax_end_element;

  //
  /*

  saxHandler.initialized = XML_SAX2_MAGIC;
  */
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void dolfin::sax_start_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs)
{
  std::cout << "start elemen" << std::endl;
  /*
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
  */
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_element(void *ctx, const xmlChar *name)
{
  /*
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
*/
}
//-----------------------------------------------------------------------------

