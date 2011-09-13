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
// Last changed: 2011-09-13

#ifndef __XML_MESH_MARKERS_H
#define __XML_MESH_MARKERS_H

#include "pugixml.hpp"
#include <dolfin/mesh/MeshValueCollection.h>
#include "XMLMesh.h"

namespace dolfin
{

  class XMLMeshValueCollection
  {
  public:

    // Read mesh markers from XML file
    template <class T>
    static void read(MeshValueCollection<T>& mesh_markers,
                     const std::string type,
                     const pugi::xml_node xml_node);

    /// Write mesh markers to XML file
    template<class T>
    static void write(const MeshValueCollection<T>& mesh_markers,
                      const std::string type,
                      pugi::xml_node xml_node,
                      bool write_mesh=true);

  };

  //---------------------------------------------------------------------------
  template <class T>
  inline void XMLMeshValueCollection::read(MeshValueCollection<T>& mesh_markers,
                                   const std::string type,
                                   const pugi::xml_node xml_node)
  {
    not_working_in_parallel("Reading XML MeshValueCollection");

    // Get mesh markers node
    const pugi::xml_node markers_node = xml_node.child("mesh_markers");
    if (!markers_node)
      error("Not a DOLFIN XML MeshValueCollection file.");

    // Get attributes
    const std::string type_file = markers_node.attribute("type").value();
    const uint dim = markers_node.attribute("dim").as_uint();
    const uint size = markers_node.attribute("size").as_uint();

    // Check that types match
    if (type != type_file)
      dolfin_error("XMLMeshValueCollection.h",
                   "Read mesh markers from XML file",
                   "Type mismatch, found \"%s\" but expecting \"%s\"",
                   type_file.c_str(), type.c_str());

    // Check that dimension matches
    if (mesh_markers.dim() != dim)
      dolfin_error("XMLMeshValueCollection.h",
                   "Read mesh markers from XML file",
                   "Dimension mismatch, found %d but expecting %d",
                   dim, mesh_markers.dim());

    // Clear old markers
    mesh_markers.clear();

    // Choose data type
    if (type == "uint")
    {
      for (pugi::xml_node_iterator it = markers_node.begin();
           it != markers_node.end(); ++it)
      {
        const uint cell_index = it->attribute("cell_index").as_uint();
        const uint local_entity = it->attribute("local_entity").as_uint();
        const uint marker_value = it->attribute("marker_value").as_uint();
        mesh_markers.set_marker(cell_index, local_entity, marker_value);
      }
    }
    else if (type == "int")
    {
      for (pugi::xml_node_iterator it = markers_node.begin();
           it != markers_node.end(); ++it)
      {
        const uint cell_index = it->attribute("cell_index").as_uint();
        const uint local_entity = it->attribute("local_entity").as_uint();
        const int marker_value = it->attribute("marker_value").as_int();
        mesh_markers.set_marker(cell_index, local_entity, marker_value);
      }
    }
    else if (type == "double")
    {
      for (pugi::xml_node_iterator it = markers_node.begin();
           it != markers_node.end(); ++it)
      {
        const uint cell_index = it->attribute("cell_index").as_uint();
        const uint local_entity = it->attribute("local_entity").as_uint();
        const double marker_value = it->attribute("marker_value").as_double();
        mesh_markers.set_marker(cell_index, local_entity, marker_value);
      }
    }
    else if (type == "bool")
    {
      for (pugi::xml_node_iterator it = markers_node.begin();
           it != markers_node.end(); ++it)
      {
        const uint cell_index = it->attribute("cell_index").as_uint();
        const uint local_entity = it->attribute("local_entity").as_uint();
        const bool marker_value = it->attribute("marker_value").as_bool();
        mesh_markers.set_marker(cell_index, local_entity, marker_value);
      }
    }
    else
      dolfin_error("XMLMarkers.h",
                   "Read mesh markers from XML file",
                   "Unhandled marker type \"%s\"", type.c_str());
  }
  //---------------------------------------------------------------------------
  template<class T>
  void XMLMeshValueCollection::write(const MeshValueCollection<T>& mesh_markers,
                             const std::string type,
                             pugi::xml_node xml_node,
                             bool write_mesh)
  {
    not_working_in_parallel("Writing XML MeshValueCollection");

    // Write mesh if requested
    if (write_mesh)
      XMLMesh::write(mesh_markers.mesh(), xml_node);

    // Add mesh function node and attributes
    pugi::xml_node mf_node = xml_node.append_child("mesh_markers");
    mf_node.append_attribute("type") = type.c_str();
    mf_node.append_attribute("dim") = mesh_markers.dim();
    mf_node.append_attribute("size") = mesh_markers.size();

    // Add data
    for (uint i = 0; i < mesh_markers.size(); ++i)
    {
      pugi::xml_node entity_node = mf_node.append_child("marker");
      const std::pair<std::pair<uint, uint>, T>& marker = mesh_markers.get_marker(i);
      entity_node.append_attribute("cell_index") = marker.first.first;
      entity_node.append_attribute("local_entity") = marker.first.second;
      entity_node.append_attribute("marker_value") = marker.second;
    }
  }
  //---------------------------------------------------------------------------

}
#endif
