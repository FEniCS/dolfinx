// Copyright (C) 2002-2011 Anders Logg, Ola Skavhaug and Garth N. Wells
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
// First added:  2002-12-06
// Last changed: 2006-10-16

#include <iomanip>
#include <iostream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>

#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/Point.h"
#include "dolfin/mesh/Vertex.h"
#include "XMLIndent.h"
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLMesh::read(Mesh& mesh, const pugi::xml_node xml_mesh)
{
  const pugi::xml_node xml_mesh_node = xml_mesh.child("mesh");
  if (!xml_mesh_node)
    error("Not a DOLFIN Mesh file.");

  // Get cell type and geometric dimension
  const std::string cell_type_str = xml_mesh_node.attribute("celltype").value();
  const unsigned int gdim = xml_mesh_node.attribute("dim").as_uint();

  // Get topological dimension
  boost::scoped_ptr<CellType> cell_type(CellType::create(cell_type_str));
  const unsigned int tdim = cell_type->dim();

  // Create mesh for editing
  MeshEditor editor;
  editor.open(mesh, cell_type_str, tdim, gdim);

  // Get vertices node
  pugi::xml_node xml_vertices = xml_mesh_node.child("vertices");
  assert(xml_vertices);

  // Get number of vertices and init editor
  const unsigned int num_vertices = xml_vertices.attribute("size").as_uint();
  editor.init_vertices(num_vertices);

  // Iterate over vertices and add to mesh
  Point p;
  for (pugi::xml_node_iterator it = xml_vertices.begin(); it != xml_vertices.end(); ++it)
  {
    const unsigned int index = it->attribute("index").as_uint();
    p[0] = it->attribute("x").as_double();
    p[1] = it->attribute("y").as_double();
    p[2] = it->attribute("z").as_double();
    editor.add_vertex(index, p);
  }

  // Get cells node
  pugi::xml_node xml_cells = xml_mesh_node.child("cells");
  assert(xml_cells);

  // Get number of cels and init editor
  const unsigned int num_cells = xml_cells.attribute("size").as_uint();
  editor.init_cells(num_cells);

  // Create list of vertex index attribute names
  const unsigned int num_vertices_per_cell = cell_type->num_vertices(tdim);
  std::vector<std::string> v_str(num_vertices_per_cell);
  for (uint i = 0; i < num_vertices_per_cell; ++i)
    v_str[i] = "v" + boost::lexical_cast<std::string, unsigned int>(i);

  // Iterate over cells and add to mesh
  std::vector<unsigned int> v(num_vertices_per_cell);
  for (pugi::xml_node_iterator it = xml_cells.begin(); it != xml_cells.end(); ++it)
  {
    const unsigned int index = it->attribute("index").as_uint();
    for (unsigned int i = 0; i < num_vertices_per_cell; ++i)
      v[i] = it->attribute(v_str[i].c_str()).as_uint();
    editor.add_cell(index, v);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
void XMLMesh::write(const Mesh& mesh, std::ostream& outfile,
                    uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Get cell type
  const CellType::Type cell_type = mesh.type().cell_type();

  // Write mesh header
  outfile << indent();
  outfile << "<mesh celltype=\"" << CellType::type2string(cell_type)
           << "\" dim=\"" << mesh.geometry().dim() << "\">" << std::endl;

  // Write vertices header
  ++indent;
  outfile << indent();
  outfile << "<vertices size=\"" << mesh.num_vertices() << "\">" << std::endl;

  // Write each vertex
  ++indent;
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    outfile << indent();
    switch (mesh.geometry().dim())
    {
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
  //++indent;
  //XMLMeshData::write(mesh.data(), outfile, indent.level());
  //--indent;

  // Write mesh footer
  --indent;
  outfile << indent() << "</mesh>" << std::endl;
}
//-----------------------------------------------------------------------------
