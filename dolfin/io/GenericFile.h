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
    virtual void read(Mesh& mesh);
    virtual void read(GenericVector& x);
    virtual void read(GenericMatrix& A);
    virtual void read(GenericDofMap& dofmap);
    virtual void read(LocalMeshData& data);
    virtual void read(MeshFunction<int>& mesh_function);
    virtual void read(MeshFunction<std::size_t>& mesh_function);
    virtual void read(MeshFunction<double>& mesh_function);
    virtual void read(MeshFunction<bool>& mesh_function);
    virtual void read(MeshValueCollection<int>& mesh_markers);
    virtual void read(MeshValueCollection<std::size_t>& mesh_markers);
    virtual void read(MeshValueCollection<double>& mesh_markers);
    virtual void read(MeshValueCollection<bool>& mesh_markers);
    virtual void read(Parameters& parameters);
    virtual void read(Table& table);
    virtual void read(std::vector<int>& x);
    virtual void read(std::vector<std::size_t>& x);
    virtual void read(std::vector<double>& x);
    virtual void read(std::map<std::size_t, int>& map);
    virtual void read(std::map<std::size_t, std::size_t>& map);
    virtual void read(std::map<std::size_t, double>& map);
    virtual void read(std::map<std::size_t, std::vector<int>>& array_map);
    virtual void read(std::map<std::size_t, std::vector<std::size_t>>& array_map);
    virtual void read(std::map<std::size_t, std::vector<double>>& array_map);
    virtual void read(Function& u);

    // Output
    virtual void write(const GenericVector& x);
    virtual void write(const GenericMatrix& A);
    virtual void write(const Mesh& mesh);
    virtual void write(const GenericDofMap& dofmap);
    virtual void write(const LocalMeshData& data);
    virtual void write(const MeshFunction<int>& mesh_function);
    virtual void write(const MeshFunction<std::size_t>& mesh_function);
    virtual void write(const MeshFunction<double>& mesh_function);
    virtual void write(const MeshFunction<bool>& mesh_function);
    virtual void write(const MeshValueCollection<int>& mesh_markers);
    virtual void write(const MeshValueCollection<std::size_t>& mesh_markers);
    virtual void write(const MeshValueCollection<double>& mesh_markers);
    virtual void write(const MeshValueCollection<bool>& mesh_markers);
    virtual void write(const Function& u);

    // Output with time
    virtual void write(const Mesh& mesh, double time);
    virtual void write(const MeshFunction<int>& mf, double time);
    virtual void write(const MeshFunction<std::size_t>& mf, double time);
    virtual void write(const MeshFunction<double>& mf, double time);
    virtual void write(const MeshFunction<bool>& mf, double time);
    virtual void write(const Function& u, double time);

    virtual void write(const Parameters& parameters);
    virtual void write(const Table& table);
    virtual void write(const std::vector<int>& x);
    virtual void write(const std::vector<std::size_t>& x);
    virtual void write(const std::vector<double>& x);
    virtual void write(const std::map<std::size_t, int>& map);
    virtual void write(const std::map<std::size_t, std::size_t>& map);
    virtual void write(const std::map<std::size_t, double>& map);
    virtual void write(const std::map<std::size_t, std::vector<int>>& array_map);
    virtual void write(const std::map<std::size_t,
                       std::vector<std::size_t>>& array_map);
    virtual void write(const std::map<std::size_t,
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
