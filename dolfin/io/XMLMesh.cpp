// Copyright (C) 2009 Anders Logg and Ola Skavhaug
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
// First added:  2009-03-06
// Last changed: 2011-03-28

#include <cstring>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include "XMLIndent.h"
#include "XMLArray.h"
#include "XMLMeshData.h"
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMesh::XMLMesh(Mesh& mesh, XMLFile& parser)
  : XMLHandler(parser), _mesh(mesh), state(OUTSIDE), f(0), a(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMesh::~XMLMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLMesh::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch (state)
  {
  case OUTSIDE:

    if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0)
    {
      read_mesh_tag(name, attrs);
    }

    break;

  case INSIDE_MESH:

    if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
    {
      read_vertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0)
    {
      read_cells(name, attrs);
      state = INSIDE_CELLS;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "data") == 0)
    {
      read_mesh_data(name, attrs);
      state = INSIDE_MESH;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "higher_order_coordinates") == 0)
    {
      read_higher_order_vertices(name, attrs);
      state = INSIDE_HIGHERORDERVERTICES;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "higher_order_cells") == 0)
    {
      read_higher_order_cells(name, attrs);
      state = INSIDE_HIGHERORDERCELLS;
    }

    break;

  case INSIDE_VERTICES:

    if (xmlStrcasecmp(name, (xmlChar *) "vertex") == 0)
      read_vertex(name, attrs);

    break;

  case INSIDE_CELLS:

    if (xmlStrcasecmp(name, (xmlChar *) "interval") == 0)
      read_interval(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "triangle") == 0)
      read_triangle(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0)
      read_tetrahedron(name, attrs);

    break;

  case INSIDE_HIGHERORDERVERTICES:

    if (xmlStrcasecmp(name, (xmlChar *) "vertex") == 0)
      read_higher_order_vertex(name, attrs);

    break;

  case INSIDE_HIGHERORDERCELLS:

    if (xmlStrcasecmp(name, (xmlChar *) "cell") == 0)
      read_higher_order_cell_data(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::end_element(const xmlChar *name)
{
  switch (state)
  {
  case INSIDE_MESH:

    if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0)
    {
      close_mesh();
      state = DONE;
      release();
    }

    break;

  case INSIDE_VERTICES:

    if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_CELLS:

    if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_HIGHERORDERVERTICES:

    if (xmlStrcasecmp(name, (xmlChar *) "higher_order_coordinates") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_HIGHERORDERCELLS:

    if (xmlStrcasecmp(name, (xmlChar *) "higher_order_cells") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write(const Mesh& mesh, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Get cell type
  CellType::Type cell_type = mesh.type().cell_type();

  // Write mesh header
  outfile << indent();
  outfile << "<mesh celltype=\"" << CellType::type2string(cell_type) << "\" dim=\"" << mesh.geometry().dim() << "\">" << std::endl;

  // Write vertices header
  ++indent;
  outfile << indent();
  outfile << "<vertices size=\"" << mesh.num_vertices() << "\">" << std::endl;

  // Write each vertex
  ++indent;
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    outfile << indent();

    switch (mesh.geometry().dim()) {
    case 1:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\"/>" << std::endl;
      break;
    case 2:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\" y=\"" << p.y() << "\"/>" << std::endl;
      break;
    case 3:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\" y=\"" << p.y()  << "\" z=\"" << p.z() << "\"/>" << std::endl;
      break;
    default:
      error("The XML mesh file format only supports 1D, 2D and 3D meshes.");
    }
  }

  // Write vertex footer
  --indent;
  outfile << indent() << "</vertices>" << std::endl;

  // Write cell header
  outfile << indent();
  outfile << "<cells size=\"" << mesh.num_cells() << "\">" << std::endl;

  // Write each cell
  ++indent;
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    const uint* vertices = c->entities(0);
    assert(vertices);
    outfile << indent();

    switch (cell_type)
    {
    case CellType::interval:
      outfile << "<interval index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\"/>" << std::endl;
      break;
    case CellType::triangle:
      outfile << "<triangle index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\" v2=\"" << vertices[2] << "\"/>" << std::endl;
      break;
    case CellType::tetrahedron:
      outfile << "<tetrahedron index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\" v2=\"" << vertices[2] << "\" v3=\"" << vertices[3] << "\"/>" << std::endl;
      break;
    default:
      error("Unknown cell type: %u.", cell_type);
    }
  }
  // Write cell footer
  --indent;
  outfile << indent() << "</cells>" << std::endl;

  // Write mesh data
  ++indent;
  XMLMeshData::write(mesh.data(), outfile, indent.level());
  --indent;

  // Write mesh footer
  --indent;
  outfile << indent() << "</mesh>" << std::endl;

}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh_tag(const xmlChar *name, const xmlChar **attrs)
{
  if (MPI::num_processes() > 1)
    warning("Reading entire mesh to one processor. If this is not what you intended, initialize the mesh directly from the filename.");

  // Set state
  state = INSIDE_MESH;

  // Parse values
  std::string type = parse_string(name, attrs, "celltype");
  uint gdim = parse_uint(name, attrs, "dim");

  // Create cell type to get topological dimension
  CellType* cell_type = CellType::create(type);
  uint tdim = cell_type->dim();
  delete cell_type;

  // Open mesh for editing
  editor.open(_mesh, CellType::string2type(type), tdim, gdim);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_vertices = parse_uint(name, attrs, "size");

  // Set number of vertices
  editor.init_vertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_cells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_cells = parse_uint(name, attrs, "size");

  // Set number of vertices
  editor.init_cells(num_cells);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parse_uint(name, attrs, "index");

  // Handle differently depending on geometric dimension
  switch (_mesh.geometry().dim())
  {
  case 1:
    {
      double x = parse_float(name, attrs, "x");
      editor.add_vertex(v, x);
    }
    break;
  case 2:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      editor.add_vertex(v, x, y);
    }
    break;
  case 3:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      double z = parse_float(name, attrs, "z");
      editor.add_vertex(v, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_interval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (_mesh.topology().dim() != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");

  // Add cell
  editor.add_cell(c, v0, v1);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_triangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (_mesh.topology().dim() != 2)
    error("Mesh entity (triangle) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");

  // Add cell
  editor.add_cell(c, v0, v1, v2);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_tetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (_mesh.topology().dim() != 3)
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");
  uint v3 = parse_uint(name, attrs, "v3");

  // Add cell
  editor.add_cell(c, v0, v1, v2, v3);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh_entity(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parse_uint(name, attrs, "index");

  // Read and set value
  assert(f);
  assert(index < f->size());
  const uint value = parse_uint(name, attrs, "value");
  (*f)[index] = value;
}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh_data(const xmlChar* name, const xmlChar** attrs)
{
  xml_mesh_data.reset(new XMLMeshData(_mesh.data(), parser, true));
  xml_mesh_data->handle();
}
//-----------------------------------------------------------------------------
void XMLMesh::read_higher_order_vertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_higher_order_vertices = parse_uint(name, attrs, "size");

  // Set number of vertices
  editor.init_higher_order_vertices(num_higher_order_vertices);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_higher_order_cells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_higher_order_cells    = parse_uint(name, attrs, "size");
  uint num_higher_order_cell_dof = parse_uint(name, attrs, "num_dof");

  // Set number of vertices
  editor.init_higher_order_cells(num_higher_order_cells, num_higher_order_cell_dof);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_higher_order_vertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parse_uint(name, attrs, "index");

  // Handle differently depending on geometric dimension
  switch (_mesh.geometry().dim())
  {
  case 1:
    {
      double x = parse_float(name, attrs, "x");
      editor.add_higher_order_vertex(v, x);
    }
    break;
  case 2:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      editor.add_higher_order_vertex(v, x, y);
    }
    break;
  case 3:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      double z = parse_float(name, attrs, "z");
      editor.add_higher_order_vertex(v, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_higher_order_cell_data(const xmlChar *name, const xmlChar **attrs)
{
  // for now assume a P2 triangle!

  // Check dimension
  if (_mesh.topology().dim() != 2)
    error("Mesh entity must be a triangle; does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  const std::string affine_str = parse_string_optional(name, attrs, "affine");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");
  uint v3 = parse_uint(name, attrs, "v3");
  uint v4 = parse_uint(name, attrs, "v4");
  uint v5 = parse_uint(name, attrs, "v5");

  // Add cell
  editor.add_higher_order_cell_data(c, v0, v1, v2, v3, v4, v5);

  // set affine indicator
  editor.set_affine_cell_indicator(c, affine_str);
}
//-----------------------------------------------------------------------------
void XMLMesh::close_mesh()
{
  editor.close(false);
}
//-----------------------------------------------------------------------------
