// Copyright (C) 2005-2009 Garth N. Wells
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
// Modified by Anders Logg 2006.
// Modified by Niclas Jansson 2009.
//
// First added:  2005-07-05
// Last changed: 2013-03-11

#ifndef __VTK_FILE_H
#define __VTK_FILE_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "GenericFile.h"

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  /// Output of meshes and functions in VTK format

  /// XML format for visualisation purposes. It is not suitable to
  /// checkpointing as it may decimate some data.

  class VTKFile : public GenericFile
  {
  public:

    /// Create VTK file
    VTKFile(const std::string filename, std::string encoding);
    ~VTKFile();

    /// Output mesh
    void operator<< (const Mesh& mesh);

    /// Output MeshFunction<bool>
    ///
    /// @param meshfunction
    void operator<< (const MeshFunction<bool>& meshfunction);

    /// Output MeshFunction<std::size_t>
    ///
    /// @param meshfunction
    void operator<< (const MeshFunction<std::size_t>& meshfunction);

    /// Output MeshFunction<int>
    ///
    /// @param meshfunction
    void operator<< (const MeshFunction<int>& meshfunction);

    /// Output MeshFunction<double>
    ///
    /// @param meshfunction
    void operator<< (const MeshFunction<double>& meshfunction);

    /// Output Function
    /// @param u (Function)
    void operator<< (const Function& u);

    /// Output Mesh and timestep
    /// @param mesh
    ///   Mesh and time
    void operator<< (const std::pair<const Mesh*, double> mesh);

    /// Output MeshFunction and timestep
    /// @param f
    ///   MeshFunction and time
    void operator<< (const std::pair<const MeshFunction<int>*, double> f);

    /// Output MeshFunction and timestep
    /// @param f
    ///   MeshFunction and time
    void
      operator<< (const std::pair<const MeshFunction<std::size_t>*, double> f);

    /// Output MeshFunction and timestep
    /// @param f
    ///   MeshFunction and time
    void operator<< (const std::pair<const MeshFunction<double>*, double> f);

    /// Output MeshFunction and timestep
    /// @param f
    ///   MeshFunction and time
    void operator<< (const std::pair<const MeshFunction<bool>*, double> f);

    /// Output Function and timestep
    /// @param u
    ///   Function and time
    void operator<< (const std::pair<const Function*, double> u);

  private:

    void write_function(const Function& u, double time);

    void write_mesh(const Mesh& mesh, double time);

    std::string init(const Mesh& mesh, std::size_t dim) const;

    void finalize(std::string vtu_filename, double time);

    void results_write(const Function& u, std::string file) const;

    void write_point_data(const GenericFunction& u, const Mesh& mesh,
                          std::string file) const;

    void pvd_file_write(std::size_t step, double time, std::string file);


    void pvtu_write_function(std::size_t dim, std::size_t rank,
                             const std::string data_location,
                             const std::string name,
                             const std::string filename,
                             std::size_t num_processes) const;

    void pvtu_write_mesh(const std::string pvtu_filename,
                         const std::size_t num_processes) const;

    void pvtu_write(const Function& u, const std::string pvtu_filename) const;

    void vtk_header_open(std::size_t num_vertices, std::size_t num_cells,
                         std::string file) const;

    void vtk_header_close(std::string file) const;

    std::string vtu_name(const int process, const int num_processes,
                         const int counter, std::string ext) const;

    void clear_file(std::string file) const;

    template<typename T>
    void mesh_function_write(T& meshfunction, double time);

    // Strip path from file
    std::string strip_path(std::string file) const;

  private:

    void pvtu_write_mesh(pugi::xml_node xml_node) const;

    // File encoding
    const std::string _encoding;
    std::string encode_string;

    bool binary;
    bool compress;

  };

}

#endif
