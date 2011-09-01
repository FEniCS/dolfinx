// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-06-30
// Last changed: 2011-09-01

#ifndef __XML_MESH_FUNCTION_H
#define __XML_MESH_FUNCTION_H

#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

#include "pugixml.hpp"
#include "dolfin/mesh/MeshFunction.h"
#include "XMLMesh.h"

namespace dolfin
{

  class XMLMeshFunction
  {
  public:

    // Read XML MeshFunction
    template <class T>
    static void read(MeshFunction<T>& mesh_function, const std::string type,
                     const pugi::xml_node xml_mesh);

    /// Write the XML file
    template<class T>
    static void write(const MeshFunction<T>& mesh_function,
                      const std::string type, pugi::xml_node xml_node,
                      bool write_mesh=true);


  };

  //---------------------------------------------------------------------------
  template <class T>
  inline void XMLMeshFunction::read(MeshFunction<T>& mesh_function,
                                    const std::string type,
                                    const pugi::xml_node xml_mesh)
  {
    not_working_in_parallel("Reading XML MeshFunctions");

    // Check for old tag
    if (xml_mesh.child("meshfunction"))
      dolfin_error("XMLMeshFunction.h",
                   "read DOLFIN MeshFunction from XML file",
                   "The XML tag <meshfunction> has been changed to <mesh_function>");

    // Read main tag
    const pugi::xml_node xml_meshfunction = xml_mesh.child("mesh_function");
    if (!xml_meshfunction)
      std::cout << "Not a DOLFIN MeshFunction." << std::endl;

    // Get type and size
    const std::string file_data_type = xml_meshfunction.attribute("type").value();
    const unsigned int dim = xml_meshfunction.attribute("dim").as_uint();
    const unsigned int size = xml_meshfunction.attribute("size").as_uint();

    // Check that types match
    if (type != file_data_type)
      error("Type mismatch reading XML MeshFunction. MeshFunction type is \"%s\", but file type is \"%s\".", file_data_type.c_str(), type.c_str());

    // Initialise MeshFunction
    mesh_function.init(dim, size);

    // Iterate over entries (choose data type)
    if (type == "uint")
    {
      for (pugi::xml_node_iterator it = xml_meshfunction.begin(); it != xml_meshfunction.end(); ++it)
      {
        const unsigned int index = it->attribute("index").as_uint();
        assert(index < size);
        mesh_function[index] = it->attribute("value").as_uint();
      }
    }
    else if (type == "int")
    {
      for (pugi::xml_node_iterator it = xml_meshfunction.begin(); it != xml_meshfunction.end(); ++it)
      {
        const unsigned int index = it->attribute("index").as_uint();
        assert(index < size);
        mesh_function[index] = it->attribute("value").as_int();
      }
    }
    else if (type == "double")
    {
      for (pugi::xml_node_iterator it = xml_meshfunction.begin(); it != xml_meshfunction.end(); ++it)
      {
        const unsigned int index = it->attribute("index").as_uint();
        assert(index < size);
        mesh_function[index] = it->attribute("value").as_double();
      }
    }
    else if (type == "bool")
    {
      for (pugi::xml_node_iterator it = xml_meshfunction.begin(); it != xml_meshfunction.end(); ++it)
      {
        const unsigned int index = it->attribute("index").as_uint();
        assert(index < size);
        mesh_function[index] = it->attribute("value").as_bool();
      }
    }
    else
      error("Type unknown in XMLMeshFunction::read.");
  }
  //---------------------------------------------------------------------------
  template<class T>
  void XMLMeshFunction::write(const MeshFunction<T>& mesh_function,
                              const std::string type, pugi::xml_node xml_node,
                              bool write_mesh)
  {
    not_working_in_parallel("Writing XML MeshFunctions");

    // Write mesh if requested
    if (write_mesh)
      XMLMesh::write(mesh_function.mesh(), xml_node);

    // Add mesh function node and attributes
    pugi::xml_node mf_node = xml_node.append_child("mesh_function");
    mf_node.append_attribute("type") = type.c_str();
    mf_node.append_attribute("dim") = mesh_function.dim();
    mf_node.append_attribute("size") = mesh_function.size();

    // Add data
    for (uint i = 0; i < mesh_function.size(); ++i)
    {
      pugi::xml_node entity_node = mf_node.append_child("entity");
      entity_node.append_attribute("index") = i;
      entity_node.append_attribute("value") = mesh_function[i];
    }
  }
  //---------------------------------------------------------------------------

}
#endif
