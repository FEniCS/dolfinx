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
#include <dolfin/mesh/LocalMeshValueCollection.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "XMLMesh.h"
#include "XMLMeshValueCollection.h"

namespace dolfin
{

  class XMLMeshFunction
  {
  public:

    // Read XML MeshFunction
    template <typename T>
    static void read(MeshFunction<T>& mesh_function, const std::string type,
                     const pugi::xml_node xml_mesh);

    // Read XML MeshFunction as a MeshValueCollection
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

    cout << "Type:  " << type << endl;

    // Read main tag
    const pugi::xml_node xml_meshfunction = xml_mesh.child(tag_name.c_str());
    if (!xml_meshfunction)
      std::cout << "Not a DOLFIN MeshFunction XML file." << std::endl;

    // Check for new (MeshValueCollection) / old storage
    if (xml_meshfunction.attributes_begin() == xml_meshfunction.attributes_end())
    {
      const Mesh& mesh = mesh_function.mesh();

      // Read new-style MeshFunction
      MeshValueCollection<T> mesh_value_collection;
      if (MPI::num_processes() == 1)
      {
        cout << "Read collection on root" << endl;
        XMLMeshValueCollection::read<T>(mesh_value_collection, type, xml_meshfunction);
        cout << "End read collection on root" << endl;
      }
      else
      {
        uint dim = 0;
        if (MPI::process_number() == 0)
        {
          XMLMeshValueCollection::read<T>(mesh_value_collection, type, xml_meshfunction);
          dim = mesh_value_collection.dim();

          const std::map<std::pair<uint, uint>, T> values = mesh_value_collection.values();  
          typename std::map<std::pair<uint, uint>, T>::const_iterator it;  
          for (it = values.begin(); it != values.end(); ++it)
          {
            cout << "Dist val(0): " << it->second << endl; 
            if (it->second < 0.2)
              error("val(0)");
          }

        }
        MPI::broadcast(dim);
        mesh_value_collection.set_dim(dim);

        // Build local data
        LocalMeshValueCollection<T> local_data(mesh_value_collection, dim);

        /*
        const std::vector<std::pair<std::pair<uint, uint>, T> >& lvalues = local_data.values();
        for (uint j = 0; j < lvalues.size(); ++j)
        {
          cout << "Local val: " << lvalues[j].second << endl; 
          if (lvalues[j].second < 0.2)
            error("local val");
        }
        */

        // Distribute MeshValueCollection
        MeshPartitioning::build_distributed_value_collection<T>(mesh_value_collection,
                                                               local_data, mesh);

        /*
        const std::map<std::pair<uint, uint>, T> values = mesh_value_collection.values();  
        typename std::map<std::pair<uint, uint>, T>::const_iterator it;  
        for (it = values.begin(); it != values.end(); ++it)
        {
          cout << "Dist val: " << it->second << endl; 
          if (it->second < 0.2)
            error("dist val");
        }
        */
      }


      // Assign collection to mesh function (this is a local operation)
      //cout << "Start assign" << endl;
      mesh_function = mesh_value_collection;
      //cout << "End assign" << endl;
      //for (uint i = 0; i < mesh_function.size(); ++i)
      //  cout << "MF: " << mesh_function[i] << ", " << i << ", " << mesh_function.size() << endl;
    }
    else
    {
      // Read old-style MeshFunction
      if (MPI::num_processes() > 1)
        error("Reading old-style XML MeshFunctions is not supported in parallel. Consider using the new format.");

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
      error("Cannot read old-style MeshFunction XML files as a MeshValueCollection.");
  }
  //---------------------------------------------------------------------------
  template<typename T>
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

    // Create MeshValueCollection for output
    const MeshValueCollection<T> mesh_value_collection(mesh_function);

    // Write MeshValueCollection
    XMLMeshValueCollection::write(mesh_value_collection, type, mf_node);
  }
  //---------------------------------------------------------------------------

}
#endif
