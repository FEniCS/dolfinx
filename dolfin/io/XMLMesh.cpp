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
// Last changed: 2011-06-30

#include <map>
#include <iomanip>
#include <iostream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshData.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/Point.h"
#include "dolfin/mesh/Vertex.h"
#include "XMLIndent.h"
#include "XMLMeshFunction.h"
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLMesh::read(Mesh& mesh, const pugi::xml_node xml_dolfin)
{
  const pugi::xml_node xml_mesh = xml_dolfin.child("mesh");
  if (!xml_mesh)
    error("Not a DOLFIN Mesh file.");

  // Read mesh
  read_mesh(mesh, xml_mesh);

  // Read any mesh data
  read_data(mesh.data(), xml_mesh);
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
  ++indent;
  write_data(mesh.data(), outfile, indent.level());
  --indent;

  // Write mesh footer
  --indent;
  outfile << indent() << "</mesh>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMesh::write(const Mesh& mesh, pugi::xml_node xml_node)
{
  // Add mesh node
  pugi::xml_node mesh_node = xml_node.append_child("mesh");

  // Add mesh attributes
  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);
  mesh_node.append_attribute("celltype") = cell_type.c_str();
  mesh_node.append_attribute("dim") = mesh.geometry().dim();

  // Add vertices node
  pugi::xml_node vertices_node = mesh_node.append_child("vertices");
  vertices_node.append_attribute("size") = mesh.num_vertices();

  // Write each vertex
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    pugi::xml_node vertex_node = vertices_node.append_child("vertex");
    vertex_node.append_attribute("index") = v->index();

    const Point p = v->point();
    switch (mesh.geometry().dim())
    {
      case 1:
        vertex_node.append_attribute("x") = p.x();
        break;
      case 2:
        vertex_node.append_attribute("x") = p.x();
        vertex_node.append_attribute("y") = p.y();
        break;
      case 3:
        vertex_node.append_attribute("x") = p.x();
        vertex_node.append_attribute("y") = p.y();
        vertex_node.append_attribute("z") = p.z();
        break;
      default:
        error("The XML mesh file format only supports 1D, 2D and 3D meshes.");
    }
  }

  // Add cells node
  pugi::xml_node cells_node = mesh_node.append_child("cells");
  cells_node.append_attribute("size") = mesh.num_cells();

  // Add each cell
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    pugi::xml_node cell_node = cells_node.append_child(cell_type.c_str());
    cell_node.append_attribute("index") = c->index();

    const uint* vertices = c->entities(0);
    assert(vertices);

    switch (_cell_type)
    {
    case CellType::interval:
      cell_node.append_attribute("v0") = vertices[0];
      cell_node.append_attribute("v1") = vertices[1];
      break;
    case CellType::triangle:
      cell_node.append_attribute("v0") = vertices[0];
      cell_node.append_attribute("v1") = vertices[1];
      cell_node.append_attribute("v2") = vertices[2];
      break;
    case CellType::tetrahedron:
      cell_node.append_attribute("v0") = vertices[0];
      cell_node.append_attribute("v1") = vertices[1];
      cell_node.append_attribute("v2") = vertices[2];
      cell_node.append_attribute("v3") = vertices[3];
      break;
    default:
      error("Unknown cell type: %u.", _cell_type);
    }
  }

  // FIXME: Write mesh data
}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh(Mesh& mesh, const pugi::xml_node xml_mesh)
{
  // Get cell type and geometric dimension
  const std::string cell_type_str = xml_mesh.attribute("celltype").value();
  const unsigned int gdim = xml_mesh.attribute("dim").as_uint();

  // Get topological dimension
  boost::scoped_ptr<CellType> cell_type(CellType::create(cell_type_str));
  const unsigned int tdim = cell_type->dim();

  // Create mesh for editing
  MeshEditor editor;
  editor.open(mesh, cell_type_str, tdim, gdim);

  // Get vertices xml node
  pugi::xml_node xml_vertices = xml_mesh.child("vertices");
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
  pugi::xml_node xml_cells = xml_mesh.child("cells");
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
void XMLMesh::read_data(MeshData& data, const pugi::xml_node xml_mesh)
{
  // Check if we have any mesh data
  const pugi::xml_node xml_data = xml_mesh.child("data");
  if (!xml_data)
    return;

  // Iterate over data
  for (pugi::xml_node_iterator it = xml_data.begin(); it != xml_data.end(); ++it)
  {
    // Check node is data_entry
    const std::string node_name = it->name();
    if (node_name != "data_entry")
      error("Expecting XML node called \"data_entry\", but go \"%s\".", node_name.c_str());

    // Get name of data set
    const std::string data_set_name = it->attribute("name").value();
    //std::cout << "MeshData name:" << data_set_name << "." << std::endl;

    // Check that there is only one data set
    if (it->first_child().next_sibling())
      error("XML file contains too many data sets.");

    // Get type of data set
    pugi::xml_node data_set = it->first_child();
    const std::string data_set_type = data_set.name();
    //std::cout << "  Data set type: " << data_set_type << std::endl;
    if (data_set_type == "array")
    {
      // Get type
      const std::string data_type = data_set.attribute("type").value();
      //std::cout << "  Data set type: " << data_type << std::endl;
      if (data_type == "uint")
      {
        // Get vector from MeshData
        std::vector<unsigned int>* array = data.array(data_set_name);
        if (!array)
          array = data.create_array(data_set_name);
        assert(array);

        // Read vector
        read_array_uint(*array, data_set);
      }
      else
        error("Only reading of MeshData uint Arrays are supported at present.");
    }
    else if (data_set_type == "meshfunction")
    {
      // Get MeshFunction from MeshData
      const std::string data_type = data_set.attribute("type").value();
      boost::shared_ptr<MeshFunction<unsigned int> > mf = data.mesh_function(data_set_name);
      if (!mf)
        mf = data.create_mesh_function(data_set_name);
      assert(mf);

      // Read  MeshFunction
      XMLMeshFunction::read(*mf, data_type, *it);
    }
    else
      error("Reading of MeshData \"%s\" not yet supported", data_set_type.c_str());
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_array_uint(std::vector<unsigned int>& array,
                              const pugi::xml_node xml_array)
{
  // Check that we have an array
  const std::string name = xml_array.name();
  if (name != "array")
    error("Expecting an XML array node.");

  // Check type is unit
  const std::string type = xml_array.attribute("type").value();
  if (type != "uint")
    error("Expecting an XML array node.");

  // Get size and resize vector
  const unsigned int size = xml_array.attribute("size").as_uint();
  array.resize(size);

  // Iterate over array entries
  for (pugi::xml_node_iterator it = xml_array.begin(); it !=xml_array.end(); ++it)
  {
    const unsigned int index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_uint();
    assert(index < size);
    array[index] = value;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write_data(const MeshData& data, std::ostream& outfile,
                         unsigned int indentation_level)
{
  //if (data.mesh_functions.size() > 0 || data.arrays.size() > 0)
  //{
    XMLIndent indent(indentation_level);

    // Write mesh data header
    outfile << indent();
    outfile << "<data>" << std::endl;

    // Increment level for data_entries
    ++indent;

    // Write mesh functions
    typedef std::map<std::string, boost::shared_ptr<MeshFunction<unsigned int> > >::const_iterator mf_iter;
    for (mf_iter it = data.mesh_functions.begin(); it != data.mesh_functions.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write mesh function (omit mesh)
      ++indent;
      XMLMeshFunction::write(*(it->second), "uint", outfile, indent.level(), false);
      --indent;

      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    typedef std::map<std::string, std::vector<uint>*>::const_iterator arr_iter;
    for (arr_iter it = data.arrays.begin(); it != data.arrays.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      assert(it->second);
      const std::vector<unsigned int>& v = *it->second;
      ++indent;
      outfile << indent() << "<array type=\"uint\" size=\"" << v.size() << "\">" << std::endl;
      ++indent;
      for (uint i = 0; i < v.size(); ++i)
        outfile << indent() << "<element index=\"" << i << "\" value=\"" << v[i] << "\"/>" << std::endl;
      --indent;
      outfile << indent() << "</array>" << std::endl;
      --indent;
      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    // Done with entries, decrement level
    --indent;

    // Write mesh data footer
    outfile << indent();
    outfile << "</data>" << std::endl;
  //}
}
//-----------------------------------------------------------------------------
