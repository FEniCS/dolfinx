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
// First added:  2003-07-15
// Last changed: 2011-08-29

#ifndef __XMLMESH_H
#define __XMLMESH_H

#include <ostream>
#include <string>
#include <vector>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class Mesh;
  class MeshData;
  class MeshDomains;

  class XMLMesh
  {
  public:

    /// Read XML vector
    static void read(Mesh& mesh, const pugi::xml_node mesh_node);

    /// Write the XML file
    static void write(const Mesh& mesh, pugi::xml_node mesh_node);

  private:

    // Read mesh
    static void read_mesh(Mesh& mesh,
                          const pugi::xml_node mesh_node);

    // Read mesh data
    static void read_data(MeshData& data,
                          const pugi::xml_node mesh_node);

    // Read mesh markers
    static void read_markers(MeshDomains& domains,
                             const pugi::xml_node mesh_node);


    // Read array
    static void read_array_uint(std::vector<unsigned int>& array,
                                const pugi::xml_node xml_array);

    // Write mesh
    static void write_mesh(const Mesh& mesh,
                           pugi::xml_node mesh_node);

    // Write mesh data
    static void write_data(const MeshData& data,
                           pugi::xml_node mesh_node);

    // Write mesh markers
    static void write_markers(const MeshDomains& domains,
                              pugi::xml_node mesh_node);

  };

}

#endif
