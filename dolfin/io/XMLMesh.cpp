// Copyright (C) 2011-2013 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2002-12-06
// Last changed: 2014-02-06

#include <map>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <boost/format.hpp>

#include "pugixml.hpp"

#include "dolfin/common/MPI.h"
#include "dolfin/common/NoDeleter.h"
#include "dolfin/geometry/Point.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/LocalMeshData.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/Vertex.h"
#include "dolfin/mesh/MeshFunction.h"
#include "XMLMeshFunction.h"
#include "XMLMeshValueCollection.h"
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLMesh::read(Mesh& mesh, const pugi::xml_node xml_dolfin)
{
  // Get mesh node
  const pugi::xml_node mesh_node = xml_dolfin.child("mesh");
  if (!mesh_node)
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh from XML file",
                 "Not a DOLFIN XML Mesh file");
  }

  // Read mesh
  read_mesh(mesh, mesh_node);

}
//-----------------------------------------------------------------------------
void XMLMesh::write(const Mesh& mesh, pugi::xml_node xml_node)
{
  // Add mesh node
  pugi::xml_node mesh_node = xml_node.append_child("mesh");

  // Write mesh
  write_mesh(mesh, mesh_node);

}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh(Mesh& mesh, const pugi::xml_node mesh_node)
{
  // Get cell type and geometric dimension
  const std::string cell_type_str = mesh_node.attribute("celltype").value();
  const std::size_t gdim = mesh_node.attribute("dim").as_uint();

  // Get topological dimension
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  const std::size_t tdim = cell_type->dim();

  // Create mesh for editing
  MeshEditor editor;
  editor.open(mesh, cell_type_str, tdim, gdim);

  // Get vertices xml node
  pugi::xml_node xml_vertices = mesh_node.child("vertices");
  dolfin_assert(xml_vertices);

  // Get number of vertices and init editor
  const std::size_t num_vertices = xml_vertices.attribute("size").as_uint();
  editor.init_vertices_global(num_vertices, num_vertices);

  // Iterate over vertices and add to mesh
  Point p;
  for (pugi::xml_node_iterator it = xml_vertices.begin();
       it != xml_vertices.end(); ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    p[0] = it->attribute("x").as_double();
    p[1] = it->attribute("y").as_double();
    p[2] = it->attribute("z").as_double();
    editor.add_vertex(index, p);
  }

  // Get cells node
  pugi::xml_node xml_cells = mesh_node.child("cells");
  dolfin_assert(xml_cells);

  // Get number of cells and init editor
  const std::size_t num_cells = xml_cells.attribute("size").as_uint();
  editor.init_cells_global(num_cells, num_cells);

  // Create list of vertex index attribute names
  const unsigned int num_vertices_per_cell = cell_type->num_vertices(tdim);
  std::vector<std::string> v_str(num_vertices_per_cell);
  for (std::size_t i = 0; i < num_vertices_per_cell; ++i)
    v_str[i] = "v" + std::to_string(i);

  // Iterate over cells and add to mesh
  std::vector<std::size_t> v(num_vertices_per_cell);
  for (pugi::xml_node_iterator it = xml_cells.begin(); it != xml_cells.end();
       ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    for (unsigned int i = 0; i < num_vertices_per_cell; ++i)
      v[i] = it->attribute(v_str[i].c_str()).as_uint();
    editor.add_cell(index, v);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
void XMLMesh::read_array_uint(std::vector<std::size_t>& array,
                              const pugi::xml_node xml_array)
{
  // Check that we have an array
  const std::string name = xml_array.name();
  if (name != "array")
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh array data from XML file",
                 "Expecting an XML array node");
  }

  // Check type is unit
  const std::string type = xml_array.attribute("type").value();
  if (type != "uint")
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh array data from XML file",
                 "Expecting an XML array node");
  }

  // Get size and resize vector
  const std::size_t size = xml_array.attribute("size").as_uint();
  array.resize(size);

  // Iterate over array entries
  for (pugi::xml_node_iterator it = xml_array.begin(); it !=xml_array.end();
       ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_uint();
    dolfin_assert(index < size);
    array[index] = value;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write_mesh(const Mesh& mesh, pugi::xml_node mesh_node)
{
  // Add mesh attributes
  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);
  mesh_node.append_attribute("celltype") = cell_type.c_str();
  mesh_node.append_attribute("dim") = (unsigned int) mesh.geometry().dim();

  // Add vertices node
  pugi::xml_node vertices_node = mesh_node.append_child("vertices");
  vertices_node.append_attribute("size") = (unsigned int) mesh.num_vertices();

  // Write each vertex
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    pugi::xml_node vertex_node = vertices_node.append_child("vertex");
    vertex_node.append_attribute("index") = (unsigned int) v->index();

    const Point p = v->point();
    switch (mesh.geometry().dim())
    {
      case 1:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        break;
      case 2:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        vertex_node.append_attribute("y")
            = boost::str(boost::format("%.15e") % p.y()).c_str();
        break;
      case 3:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        vertex_node.append_attribute("y")
            = boost::str(boost::format("%.15e") % p.y()).c_str();
        vertex_node.append_attribute("z")
            = boost::str(boost::format("%.15e") % p.z()).c_str();
        break;
      default:
        dolfin_error("XMLMesh.cpp",
                     "write mesh to XML file",
                     "The XML mesh file format only supports 1D, 2D and 3D meshes");
    }
  }

  // Add cells node
  pugi::xml_node cells_node = mesh_node.append_child("cells");
  cells_node.append_attribute("size") = (unsigned int) mesh.num_cells();

  // Add each cell
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    pugi::xml_node cell_node = cells_node.append_child(cell_type.c_str());
    cell_node.append_attribute("index") = (unsigned int) c->index();

    const unsigned int* vertices = c->entities(0);
    dolfin_assert(vertices);

    switch (_cell_type)
    {
    case CellType::Type::interval:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      break;
    case CellType::Type::triangle:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      cell_node.append_attribute("v2") = (unsigned int) vertices[2];
      break;
    case CellType::Type::tetrahedron:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      cell_node.append_attribute("v2") = (unsigned int) vertices[2];
      cell_node.append_attribute("v3") = (unsigned int) vertices[3];
      break;
    default:
      dolfin_error("XMLMesh.cpp",
                   "write mesh to XML file",
                   "Unknown cell type (%u)", _cell_type);
    }
  }
}
//-----------------------------------------------------------------------------

