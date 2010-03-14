// Copyright (C) 2005-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
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

  protected:

    std::string init(const Mesh& mesh) const;

    void finalize(std::string vtu_filename);

    void mesh_write(const Mesh& mesh, std::string file) const;

    void results_write(const Function& u, std::string file) const;

    void write_point_data(const GenericFunction& u, const Mesh& mesh,
                          std::string file) const;

    void write_cell_data(const Function& u, std::string file) const;

    void pvd_file_write(uint u, std::string file);

    void pvtu_mesh_write(std::string pvtu_filename, std::string vtu_filename) const;

    void pvtu_results_write(const Function& u, std::string pvtu_filename) const;

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

    // Compute base64 encoded stream for VTK
    template<typename T>
    void encode_stream(std::stringstream& stream, const std::vector<T>& data) const;

  private:

    // Compute base64 encoded stream for VTK
    template<typename T>
    void encode_inline_base64(std::stringstream& stream, const std::vector<T>& data) const;

    // Compute compressed base64 encoded stream for VTK
    template<typename T>
    void encode_inline_compressed_base64(std::stringstream& stream, const std::vector<T>& data) const;

    // Most recent position in pvd file
    std::ios::pos_type mark;

    // File encoding
    const std::string encoding;

    std::string encode_string;
  };

}

#endif
