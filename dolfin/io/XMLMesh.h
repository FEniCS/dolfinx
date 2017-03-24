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
// First added:  2003-07-15
// Last changed: 2011-09-02

#ifndef __XML_MESH_H
#define __XML_MESH_H

#include <ostream>
#include <string>
#include <vector>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class LocalMeshData;
  class Mesh;
  class MeshData;
  class MeshDomains;

  /// I/O of XML representation of a Mesh

  class XMLMesh
  {
  public:

    /// Read mesh from XML
    static void read(Mesh& mesh, const pugi::xml_node mesh_node);

    /// Write mesh to XML
    static void write(const Mesh& mesh, pugi::xml_node mesh_node);

  private:

    // Read mesh
    static void read_mesh(Mesh& mesh,
                          const pugi::xml_node mesh_node);

    // Read mesh data
    static void read_data(MeshData& data,
                          const Mesh& mesh,
                          const pugi::xml_node mesh_node);

    // Read mesh domains
    static void read_domains(MeshDomains& domains,
                             const Mesh& mesh,
                             const pugi::xml_node mesh_node);

  public:

    // FIXME: This is hack for domain data support via XML in
    // parallel.
    /// Read domain data in LocalMeshData.
    static void read_domain_data(LocalMeshData& mesh_data,
                                 const pugi::xml_node mesh_node);

  private:

    // Read array
    static void read_array_uint(std::vector<std::size_t>& array,
                                const pugi::xml_node xml_array);

    // Write mesh
    static void write_mesh(const Mesh& mesh,
                           pugi::xml_node mesh_node);

    // Write mesh data
    static void write_data(const Mesh& mesh, const MeshData& data,
                           pugi::xml_node mesh_node);

    // Write mesh markers
    static void write_domains(const Mesh& mesh,
                              const MeshDomains& domains,
                              pugi::xml_node mesh_node);

  };

}

#endif
