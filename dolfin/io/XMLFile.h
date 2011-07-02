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
// Modified by Anders Logg, 2011.
//
// First added:  2009-03-03
// Last changed: 2011-06-30

#ifndef __XMLFILE_H
#define __XMLFILE_H

#include <ostream>
#include <string>
#include <boost/shared_ptr.hpp>
#include "GenericFile.h"

namespace pugi
{
  class xml_document;
  class xml_node;
}

namespace dolfin
{

  class GenericVector;
  class LocalMeshData;
  class Mesh;
  class Parameters;
  template<class T> class MeshFunction;

  class XMLFile: public GenericFile
  {
  public:

    /// Constructor
    XMLFile(const std::string filename);

    /// Constructor from a stream
    XMLFile(std::ostream& s);

    ~XMLFile();

    // Mesh
    void operator>> (Mesh& input);
    void operator<< (const Mesh& output);

    // LocalMeshData
    void operator>> (LocalMeshData& input);
    void operator<< (const LocalMeshData& output);

    // Vector
    void operator>> (GenericVector& input);
    void operator<< (const GenericVector& output);

    // Parameters
    void operator>> (Parameters& input);
    void operator<< (const Parameters& output);

    // FunctionPlotData
    void operator>> (FunctionPlotData& input);
    void operator<< (const FunctionPlotData& output);

    void operator>> (MeshFunction<int>& input)
    { read_mesh_function(input, "int"); }
    void operator<< (const MeshFunction<int>& output)
    { write_mesh_function(output, "int"); }

    void operator>> (MeshFunction<unsigned int>& input)
    { read_mesh_function(input, "uint"); }
    void operator<< (const MeshFunction<unsigned int>& output)
    { write_mesh_function(output, "uint"); }

    void operator>> (MeshFunction<double>& input)
    { read_mesh_function(input, "double"); }
    void operator<< (const MeshFunction<double>& output)
    { write_mesh_function(output, "double"); }

    void operator>> (MeshFunction<bool>& input)
    { read_mesh_function(input, "bool"); }
    void operator<< (const MeshFunction<bool>& input)
    { write_mesh_function(input, "bool"); }

  private:

    // Write MeshFunction
    template<class T> void read_mesh_function(MeshFunction<T>& t,
                                              const std::string type) const;

    // Read MeshFunction
    template<class T> void write_mesh_function(const MeshFunction<T>& t,
                                               const std::string type);

    // Get DOLFIN XML node
    const pugi::xml_node get_dolfin_xml_node(pugi::xml_document& xml_doc,
                                             const std::string filename) const;

    static pugi::xml_node write_dolfin(pugi::xml_document& doc);

    boost::shared_ptr<std::ostream> outstream;

  };

}
#endif
