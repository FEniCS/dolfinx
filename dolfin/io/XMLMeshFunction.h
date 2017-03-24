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
// Last changed: 2011-11-14

#ifndef __XML_MESH_FUNCTION_H
#define __XML_MESH_FUNCTION_H

#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

#include "pugixml.hpp"
#include <dolfin/mesh/LocalMeshValueCollection.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include "XMLMesh.h"
#include "XMLMeshValueCollection.h"

#include <dolfin/common/Timer.h>

namespace dolfin
{

  /// I/O of XML representation of MeshFunction

  class XMLMeshFunction
  {
  public:

    /// Read XML MeshFunction
    template <typename T>
    static void read(MeshFunction<T>& mesh_function, const std::string type,
                     const pugi::xml_node xml_mesh);

    /// Read XML MeshFunction as a MeshValueCollection
    template <typename T>
    static void read(MeshValueCollection<T>& mesh_value_collection,
                     const std::string type, const pugi::xml_node xml_mesh);

    /// Write the XML file
    template<typename T>
    static void write(const MeshFunction<T>& mesh_function,
                      const std::string type, pugi::xml_node xml_node,
                      bool write_mesh=true);

  };

  //---------------------------------------------------------------------------
  template <typename T>
  inline void XMLMeshFunction::read(MeshFunction<T>& mesh_function,
                                    const std::string type,
                                    const pugi::xml_node xml_mesh)
  {
    bool new_format = false;
    pugi::xml_node xml_meshfunction;
    if (std::string(xml_mesh.name()) == "mesh_function")
    {
      // XML node is a mesh function
      new_format = true;
      xml_meshfunction = xml_mesh;
    }
    else
    {
      // Tag
      std::string tag_name = "mesh_function";

      // Check for old tag
      if (xml_mesh.child("meshfunction"))
      {
        warning("The XML tag <meshfunction> has been changed to <mesh_function>. "
                "I'll be nice and read your XML data anyway, for now, but you will "
                "need to update your XML files (a simple search and replace) to use "
                "future versions of DOLFIN.");
        tag_name = "meshfunction";
      }

      // Get XML node
      xml_meshfunction = xml_mesh.child(tag_name.c_str());

      // Read main tag
      if (!xml_meshfunction)
        std::cout << "Not a DOLFIN MeshFunction XML file." << std::endl;

      if (xml_meshfunction.attributes_begin() == xml_meshfunction.attributes_end())
        new_format = true;
    }

    // Check for new (MeshValueCollection) / old storage
    if (new_format)
    {
      // Read new-style MeshFunction
      MeshValueCollection<T> mesh_value_collection(mesh_function.mesh());
      XMLMeshValueCollection::read<T>(mesh_value_collection, type,
                                      xml_meshfunction);

      // Assign collection to mesh function (this is a local
      // operation) and attach name
      mesh_function = mesh_value_collection;
      mesh_function.rename(mesh_value_collection.name(),
                           mesh_value_collection.label());
    }
    else
    {
      // Get type and size
      const std::string file_data_type = xml_meshfunction.attribute("type").value();
      const std::size_t dim = xml_meshfunction.attribute("dim").as_uint();
      const std::size_t size = xml_meshfunction.attribute("size").as_uint();

      // Check that types match
      if (type != file_data_type)
      {
        dolfin_error("XMLMeshFunction.h",
                     "read mesh function from XML file",
                     "Type mismatch reading XML MeshFunction. MeshFunction type is \"%s\", but file type is \"%s\"",
                     type.c_str(), file_data_type.c_str());
      }

      // Initialise MeshFunction
      mesh_function.init(dim, size);

      // Iterate over entries (choose data type)
      if (type == "uint")
      {
        for (pugi::xml_node_iterator it = xml_meshfunction.begin();
             it != xml_meshfunction.end(); ++it)
        {
          const std::size_t index = it->attribute("index").as_uint();
          dolfin_assert(index < size);
          mesh_function[index] = it->attribute("value").as_uint();
        }
      }
      else if (type == "int")
      {
        for (pugi::xml_node_iterator it = xml_meshfunction.begin();
             it != xml_meshfunction.end(); ++it)
        {
          const std::size_t index = it->attribute("index").as_uint();
          dolfin_assert(index < size);
          mesh_function[index] = it->attribute("value").as_int();
        }
      }
      else if (type == "double")
      {
        for (pugi::xml_node_iterator it = xml_meshfunction.begin();
             it != xml_meshfunction.end(); ++it)
        {
          const std::size_t index = it->attribute("index").as_uint();
          dolfin_assert(index < size);
          mesh_function[index] = it->attribute("value").as_double();
        }
      }
      else if (type == "bool")
      {
        for (pugi::xml_node_iterator it = xml_meshfunction.begin();
             it != xml_meshfunction.end(); ++it)
        {
          const std::size_t index = it->attribute("index").as_uint();
          dolfin_assert(index < size);
          mesh_function[index] = it->attribute("value").as_bool();
        }
      }
      else
      {
        dolfin_error("XMLMeshFunction.h",
                     "read mesh function from XML file",
                     "Unknown value type (\"%s\")", type.c_str());
      }
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  inline void XMLMeshFunction::read(MeshValueCollection<T>& mesh_value_collection,
                                    const std::string type,
                                    const pugi::xml_node xml_mesh)
  {
    // Check for old tag
    std::string tag_name("mesh_function");
    if (xml_mesh.child("meshfunction"))
    {
      warning("The XML tag <meshfunction> has been changed to <mesh_function>. "
              "I'll be nice and read your XML data anyway, for now, but you will "
              "need to update your XML files (a simple search and replace) to use "
              "future versions of DOLFIN.");
      tag_name = "meshfunction";
    }

    // Read main tag
    const pugi::xml_node xml_meshfunction = xml_mesh.child(tag_name.c_str());
    if (!xml_meshfunction)
      std::cout << "Not a DOLFIN MeshFunction XML file." << std::endl;

    // Check for new (MeshValueCollection) / old storage
    if (xml_meshfunction.attributes_begin() == xml_meshfunction.attributes_end())
    {
      // Read new-style MeshFunction
      XMLMeshValueCollection::read<T>(mesh_value_collection, type,
                                      xml_meshfunction);
    }
    else
    {
      dolfin_error("XMLMeshFunction.h",
                   "read mesh function from XML file",
                   "Cannot read old-style MeshFunction XML files as a MeshValueCollection");
    }
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void XMLMeshFunction::write(const MeshFunction<T>& mesh_function,
                              const std::string type, pugi::xml_node xml_node,
                              bool write_mesh)
  {
    not_working_in_parallel("MeshFunction XML output");

    // Write mesh if requested
    if (write_mesh)
      XMLMesh::write(*mesh_function.mesh(), xml_node);

    // Add mesh function node and attributes
    pugi::xml_node mf_node = xml_node.append_child("mesh_function");

    // Create MeshValueCollection for output
    MeshValueCollection<T> mesh_value_collection(mesh_function);
    mesh_value_collection.rename(mesh_function.name(), mesh_function.label());

    // Write MeshValueCollection
    XMLMeshValueCollection::write(mesh_value_collection, type, mf_node);
  }
  //---------------------------------------------------------------------------

}
#endif
