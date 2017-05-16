// Copyright (C) 2003-2011 Johan Hoffman and Anders Logg
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
// Modified by Ola Skavhaug 2009.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

  class GenericDofMap;
  class Function;
  class GenericMatrix;
  class GenericVector;
  class LocalMeshData;
  class Mesh;
  template <typename T> class MeshFunction;
  template <typename T> class MeshValueCollection;
  class Parameters;
  class Table;

  /// Base class for file I/O objects

  class GenericFile
  {
  public:

    /// Constructor
    GenericFile(std::string filename,
                std::string filetype);

    /// Destructor
    virtual ~GenericFile();

    // Input
    virtual void operator>> (Mesh& mesh);
    virtual void operator>> (GenericVector& x);
    virtual void operator>> (GenericMatrix& A);
    virtual void operator>> (GenericDofMap& dofmap);
    virtual void operator>> (LocalMeshData& data);
    virtual void operator>> (MeshFunction<int>& mesh_function);
    virtual void operator>> (MeshFunction<std::size_t>& mesh_function);
    virtual void operator>> (MeshFunction<double>& mesh_function);
    virtual void operator>> (MeshFunction<bool>& mesh_function);
    virtual void operator>> (MeshValueCollection<int>& mesh_markers);
    virtual void operator>> (MeshValueCollection<std::size_t>& mesh_markers);
    virtual void operator>> (MeshValueCollection<double>& mesh_markers);
    virtual void operator>> (MeshValueCollection<bool>& mesh_markers);
    virtual void operator>> (Parameters& parameters);
    virtual void operator>> (Table& table);
    virtual void operator>> (std::vector<int>& x);
    virtual void operator>> (std::vector<std::size_t>& x);
    virtual void operator>> (std::vector<double>& x);
    virtual void operator>> (std::map<std::size_t, int>& map);
    virtual void operator>> (std::map<std::size_t, std::size_t>& map);
    virtual void operator>> (std::map<std::size_t, double>& map);
    virtual void operator>> (std::map<std::size_t, std::vector<int>>& array_map);
    virtual void operator>> (std::map<std::size_t, std::vector<std::size_t>>& array_map);
    virtual void operator>> (std::map<std::size_t, std::vector<double>>& array_map);
    virtual void operator>> (Function& u);

    // Output
    virtual void operator<< (const GenericVector& x);
    virtual void operator<< (const GenericMatrix& A);
    virtual void operator<< (const Mesh& mesh);
    virtual void operator<< (const GenericDofMap& dofmap);
    virtual void operator<< (const LocalMeshData& data);
    virtual void operator<< (const MeshFunction<int>& mesh_function);
    virtual void operator<< (const MeshFunction<std::size_t>& mesh_function);
    virtual void operator<< (const MeshFunction<double>& mesh_function);
    virtual void operator<< (const MeshFunction<bool>& mesh_function);
    virtual void operator<< (const MeshValueCollection<int>& mesh_markers);
    virtual void operator<< (const MeshValueCollection<std::size_t>& mesh_markers);
    virtual void operator<< (const MeshValueCollection<double>& mesh_markers);
    virtual void operator<< (const MeshValueCollection<bool>& mesh_markers);
    virtual void operator<< (const Function& u);

    // Output with time
    virtual void operator<< (const std::pair<const Mesh*, double> mesh);
    virtual void operator<< (const std::pair<const MeshFunction<int>*, double> f);
    virtual void operator<< (const std::pair<const MeshFunction<std::size_t>*, double> f);
    virtual void operator<< (const std::pair<const MeshFunction<double>*, double> f);
    virtual void operator<< (const std::pair<const MeshFunction<bool>*, double> f);
    virtual void operator<< (const std::pair<const Function*, double> u);

    virtual void operator<< (const Parameters& parameters);
    virtual void operator<< (const Table& table);
    virtual void operator<< (const std::vector<int>& x);
    virtual void operator<< (const std::vector<std::size_t>& x);
    virtual void operator<< (const std::vector<double>& x);
    virtual void operator<< (const std::map<std::size_t, int>& map);
    virtual void operator<< (const std::map<std::size_t, std::size_t>& map);
    virtual void operator<< (const std::map<std::size_t, double>& map);
    virtual void operator<< (const std::map<std::size_t, std::vector<int>>& array_map);
    virtual void operator<< (const std::map<std::size_t,
                             std::vector<std::size_t>>& array_map);
    virtual void operator<< (const std::map<std::size_t,
                             std::vector<double>>& array_map);

    void _read();
    void _write(std::size_t process_number);

    // Return filename
    std::string name() const
    { return _filename; }

  protected:

    void read_not_impl(const std::string object) const;
    void write_not_impl(const std::string object) const;

    std::string _filename;
    std::string _filetype;

    bool opened_read;
    bool opened_write;

    // True if we have written a header
    bool check_header;

    // Counters for the number of times various data has been written
    std::size_t counter;
    std::size_t counter1;
    std::size_t counter2;

  };

}

#endif
