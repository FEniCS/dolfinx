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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2006.
// Modified by NIclas Jansson 2009.
//
// First added:  2005-07-05
// Last changed: 2009-08-13

#ifndef __VTK_FILE_H
#define __VTK_FILE_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "GenericFile.h"

namespace dolfin
{

  class VTKFile : public GenericFile
  {
  public:

    VTKFile(const std::string filename, std::string encoding);
    ~VTKFile();

    void operator<< (const Mesh& mesh);
    void operator<< (const MeshFunction<bool>& meshfunction);
    void operator<< (const MeshFunction<unsigned int>& meshfunction);
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);
    void operator<< (const Function& u);
    void operator<< (const std::pair<const Function*, double> u);

  protected:

    void write(const Function& u, double time);

    std::string init(const Mesh& mesh, uint dim) const;

    void finalize(std::string vtu_filename, double time);

    void results_write(const Function& u, std::string file) const;

    void write_point_data(const GenericFunction& u, const Mesh& mesh,
                          std::string file) const;

    void pvd_file_write(uint step, double time, std::string file);

    void pvtu_mesh_write(std::string pvtu_filename, std::string vtu_filename) const;

    void pvtu_results_write(const Function& u, std::string pvtu_filename) const;

    void pvtu_results_write(uint dim, uint rank, std::string data_type,
                            std::string name, std::string pvtu_filename) const;

    void vtk_header_open(uint num_vertices, uint num_cells, std::string file) const;
    void vtk_header_close(std::string file) const;

    void pvtu_header_open(std::string pvtu_filename) const;
    void pvtu_header_close(std::string pvtu_filename) const;

    std::string vtu_name(const int process, const int num_processes,
                         const int counter, std::string ext) const;

    void clear_file(std::string file) const;

    template<class T>
    void mesh_function_write(T& meshfunction);

    // Strip path from file
    std::string strip_path(std::string file) const;

  private:

    // Most recent position in pvd file
    std::ios::pos_type mark;

    // File encoding
    const std::string encoding;
    std::string encode_string;

    bool binary;
    bool compress;

  };

}

#endif
