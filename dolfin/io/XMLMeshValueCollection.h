// Copyright (C) 2011 Anders Logg
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
// First added:  2011-06-30
// Last changed: 2011-11-15

#ifndef __XML_MESH_VALUE_COLLECTION_H
#define __XML_MESH_VALUE_COLLECTION_H

#include <string>
#include <dolfin/mesh/MeshValueCollection.h>
#include "pugixml.hpp"
#include "xmlutils.h"
#include "XMLMesh.h"

namespace dolfin
{

  class XMLMeshValueCollection
  {
  public:

    // Read mesh value collection from XML file
    template <typename T>
    static void read(MeshValueCollection<T>& mesh_value_collection,
                     const std::string type,
                     const pugi::xml_node xml_node);

    /// Write mesh value collection to XML file
    template<typename T>
    static void write(const MeshValueCollection<T>& mesh_value_collection,
                      const std::string type,
                      pugi::xml_node xml_node);

  };

  //---------------------------------------------------------------------------
  template <typename T>
    void
    XMLMeshValueCollection::read(MeshValueCollection<T>& mesh_value_collection,
                                 const std::string type,
                                 const pugi::xml_node xml_node)
  {
    // Get node
    const pugi::xml_node mvc_node
      = xmlutils::get_node(xml_node, "mesh_value_collection");
    dolfin_assert(mvc_node);

    // Get attributes
    const std::string name = mvc_node.attribute("name").value();
    const std::string type_file = mvc_node.attribute("type").value();
    const std::size_t dim = mvc_node.attribute("dim").as_uint();

    // Attach name to mesh value collection object
    mesh_value_collection.rename(name, "a mesh value collection");

    // Set dimension
    mesh_value_collection.init(dim);

    // Check that types match
    if (type != type_file)
    {
      dolfin_error("XMLMeshValueCollection.h",
                   "read mesh value collection from XML file",
                   "Type mismatch, found \"%s\" but expecting \"%s\"",
                   type_file.c_str(), type.c_str());
    }

    // Clear old values
    mesh_value_collection.clear();

    // Choose data type
    if (type == "uint")
    {
      pugi::xml_node_iterator it;
      for (it = mvc_node.begin(); it != mvc_node.end(); ++it)
      {
        const std::size_t cell_index = it->attribute("cell_index").as_uint();
        const std::size_t local_entity
          = it->attribute("local_entity").as_uint();
        const std::size_t value = it->attribute("value").as_uint();
        mesh_value_collection.set_value(cell_index, local_entity, value);
      }
    }
    else if (type == "int")
    {
      pugi::xml_node_iterator it;
      for (it = mvc_node.begin(); it != mvc_node.end(); ++it)
      {
        const std::size_t cell_index = it->attribute("cell_index").as_uint();
        const std::size_t local_entity
          = it->attribute("local_entity").as_uint();
        const int value = it->attribute("value").as_int();
        mesh_value_collection.set_value(cell_index, local_entity, value);
      }
    }
    else if (type == "double")
    {
      pugi::xml_node_iterator it;
      for (it = mvc_node.begin(); it != mvc_node.end(); ++it)
      {
        const std::size_t cell_index = it->attribute("cell_index").as_uint();
        const std::size_t local_entity
          = it->attribute("local_entity").as_uint();
        const double value = it->attribute("value").as_double();
        mesh_value_collection.set_value(cell_index, local_entity, value);
      }
    }
    else if (type == "bool")
    {
      pugi::xml_node_iterator it;
      for (it = mvc_node.begin(); it != mvc_node.end(); ++it)
      {
        const std::size_t cell_index = it->attribute("cell_index").as_uint();
        const std::size_t local_entity
          = it->attribute("local_entity").as_uint();
        const bool value = it->attribute("value").as_bool();
        mesh_value_collection.set_value(cell_index, local_entity, value);
      }
    }
    else
    {
      dolfin_error("XMLValueCollection.h",
                   "read mesh value collection from XML file",
                   "Unhandled value type \"%s\"", type.c_str());
    }
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void XMLMeshValueCollection::write(const MeshValueCollection<T>&
                                       mesh_value_collection,
                                     const std::string type,
                                     pugi::xml_node xml_node)
  {
    not_working_in_parallel("Writing XML MeshValueCollection");

    // Add mesh function node and attributes
    pugi::xml_node mf_node = xml_node.append_child("mesh_value_collection");
    mf_node.append_attribute("name") = mesh_value_collection.name().c_str();
    mf_node.append_attribute("type") = type.c_str();
    mf_node.append_attribute("dim")
      = (unsigned int)mesh_value_collection.dim();
    mf_node.append_attribute("size")
      = (unsigned int) mesh_value_collection.size();

    // Add data
    const std::map<std::pair<std::size_t, std::size_t>, T>&
      values = mesh_value_collection.values();
    typename std::map<std::pair<std::size_t,
      std::size_t>, T>::const_iterator it;
    for (it = values.begin(); it != values.end(); ++it)
    {
      pugi::xml_node entity_node = mf_node.append_child("value");
      entity_node.append_attribute("cell_index")
        = (unsigned int) it->first.first;
      entity_node.append_attribute("local_entity")
        = (unsigned int) it->first.second;
      entity_node.append_attribute("value")
        = std::to_string(it->second).c_str();
    }
  }
  //---------------------------------------------------------------------------

}
#endif
