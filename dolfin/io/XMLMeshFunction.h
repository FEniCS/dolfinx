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
// Last changed: 2006-05-23

#ifndef __XMLMESHFUNCTION_H
#define __XMLMESHFUNCTION_H

#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

#include "pugixml.hpp"

#include "dolfin/mesh/MeshFunction.h"
#include "XMLIndent.h"
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

    /// Write the XML MeshFunction
    template <class T>
    static void write(const MeshFunction<T>& mesh_function,
                      const std::string type,
                      std::ostream& outfile, unsigned int indentation_level=0,
                      bool write_mesh=true);

  };

  //---------------------------------------------------------------------------
  template <class T>
  inline void XMLMeshFunction::read(MeshFunction<T>& mesh_function,
                                    const std::string type,
                                    const pugi::xml_node xml_mesh)
  {
    const pugi::xml_node xml_meshfunction = xml_mesh.child("meshfunction");
    if (!xml_meshfunction)
      std::cout << "Not a DOLFIN MeshFunction." << std::endl;

    // Get type and size
    const std::string data_type  = xml_meshfunction.attribute("type").value();
    const unsigned int dim = xml_meshfunction.attribute("dim").as_uint();
    const unsigned int size = xml_meshfunction.attribute("size").as_uint();

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
    else
      error("Type unknown in XMLMeshFunction::read.");
  }
  //---------------------------------------------------------------------------
  template <class T>
  inline void XMLMeshFunction::write(const MeshFunction<T>& mf,
                                     const std::string type,
                                     std::ostream& outfile,
                                     unsigned int indentation_level,
                                     bool write_mesh)
  {
    // Write mesh if requested
    if (write_mesh)
      XMLMesh::write(mf.mesh(), outfile, indentation_level);

    // Write MeshFunction
    XMLIndent indent(indentation_level);
    outfile << indent();
    outfile << "<meshfunction type=\"" << type << "\" dim=\"" << mf.dim() << "\" size=\"" << mf.size() << "\">" << std::endl;

    ++indent;
    for (uint i = 0; i < mf.size(); ++i)
    {
      outfile << indent();
      outfile << "<entity index=\"" << i << "\" value=\"" << mf[i] << "\"/>" << std::endl;
    }
    --indent;
    outfile << indent() << "</meshfunction>" << std::endl;
  }
  //---------------------------------------------------------------------------

}
#endif
