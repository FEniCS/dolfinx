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
// Last changed: 2011-09-17

#ifndef __XMLFILE_H
#define __XMLFILE_H

#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "GenericFile.h"

namespace pugi
{
  class xml_document;
  class xml_node;
}

namespace dolfin
{

  class Function;
  class GenericVector;
  class LocalMeshData;
  class Mesh;
  class Parameters;
  template<typename T> class Array;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;

  class XMLFile : public GenericFile
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
    void read_vector(std::vector<double>& input, std::vector<DolfinIndex>& indices);
    void operator<< (const GenericVector& output);

    // Parameters
    void operator>> (Parameters& input);
    void operator<< (const Parameters& output);

    // Function data
    void operator>>(Function& input);
    void operator<<(const Function& output);

    // MeshFunction (uint)
    void operator>> (MeshFunction<unsigned int>& input)
    { read_mesh_function(input, "uint"); }
    void operator<< (const MeshFunction<unsigned int>& output)
    { write_mesh_function(output, "uint"); }

    // MeshFunction (uint)
    void operator>> (MeshFunction<unsigned long int>& input)
    { read_mesh_function(input, "uint"); }
    void operator<< (const MeshFunction<unsigned long int>& output)
    { write_mesh_function(output, "uint"); }

    // MeshFunction (int)
    void operator>> (MeshFunction<int>& input)
    { read_mesh_function(input, "int"); }
    void operator<< (const MeshFunction<int>& output)
    { write_mesh_function(output, "int"); }

    // MeshFunction (double)
    void operator>> (MeshFunction<double>& input)
    { read_mesh_function(input, "double"); }
    void operator<< (const MeshFunction<double>& output)
    { write_mesh_function(output, "double"); }

    // MeshFunction (bool)
    void operator>> (MeshFunction<bool>& input)
    { read_mesh_function(input, "bool"); }
    void operator<< (const MeshFunction<bool>& input)
    { write_mesh_function(input, "bool"); }

    // MeshValueCollection (unsigned int)
    void operator>> (MeshValueCollection<unsigned int>& input)
    { read_mesh_value_collection(input, "uint"); }
    void operator<< (const MeshValueCollection<unsigned int>& output)
    { write_mesh_value_collection(output, "uint"); }

    // MeshValueCollection (unsigned long int)
    void operator>> (MeshValueCollection<unsigned long int>& input)
    { read_mesh_value_collection(input, "uint"); }
    void operator<< (const MeshValueCollection<unsigned long int>& output)
    { write_mesh_value_collection(output, "uint"); }

    // MeshValueCollection (int)
    void operator>> (MeshValueCollection<int>& input)
    { read_mesh_value_collection(input, "int"); }
    void operator<< (const MeshValueCollection<int>& output)
    { write_mesh_value_collection(output, "int"); }

    // MeshValueCollection (double)
    void operator>> (MeshValueCollection<double>& input)
    { read_mesh_value_collection(input, "double"); }
    void operator<< (const MeshValueCollection<double>& output)
    { write_mesh_value_collection(output, "double"); }

    // MeshValueCollection (bool)
    void operator>> (MeshValueCollection<bool>& input)
    { read_mesh_value_collection(input, "bool"); }
    void operator<< (const MeshValueCollection<bool>& input)
    { write_mesh_value_collection(input, "bool"); }

  private:

    // Read MeshFunction
    template<typename T> void read_mesh_function(MeshFunction<T>& t,
                                                 const std::string type) const;

    // Write MeshFunction
    template<typename T> void write_mesh_function(const MeshFunction<T>& t,
                                                  const std::string type);

    // Read MeshValueCollection
    template<typename T>
    void read_mesh_value_collection(MeshValueCollection<T>& t,
                                    const std::string type) const;

    // Write MeshValueCollection
    template<typename T>
    void write_mesh_value_collection(const MeshValueCollection<T>& t,
                                     const std::string type);

    // Load/open XML doc (from file)
    void load_xml_doc(pugi::xml_document& xml_doc) const;

    // Save XML doc (to file or stream)
    void save_xml_doc(const pugi::xml_document& xml_doc) const;

    // Get DOLFIN XML node
    const pugi::xml_node get_dolfin_xml_node(pugi::xml_document& xml_doc) const;

    static pugi::xml_node write_dolfin(pugi::xml_document& doc);

    boost::shared_ptr<std::ostream> outstream;

  };

}
#endif
