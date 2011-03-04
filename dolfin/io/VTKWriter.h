// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-07-19
// Last changed:

#ifndef __VTK_WRITER_H
#define __VTK_WRITER_H

#include <string>
#include <vector>

namespace dolfin
{

  class Function;
  class Mesh;

  class VTKWriter
  {
  public:

    // Mesh writer
    static void write_mesh(const Mesh& mesh, uint cell_dim, std::string file,
                           bool binary, bool compress);

    // Cell data writer
    static void write_cell_data(const Function& u, std::string file,
                                bool binary, bool compress);

    // Form (compressed) base64 encoded string for VTK
    template<typename T>
    static std::string encode_stream(const std::vector<T>& data,
                                     bool compress);

  private:

    // Cell data
    static std::string ascii_cell_data(const Mesh& mesh,
                                       const std::vector<uint>& offset,
                                       const std::vector<double>& values,
                                       uint dim, uint rank);
    static std::string base64_cell_data(const Mesh& mesh,
                                        const std::vector<uint>& offset,
                                        const std::vector<double>& values,
                                        uint dim, uint rank, bool compress);

    // Mesh writer
    static void write_ascii_mesh(const Mesh& mesh, uint cell_dim,
                                 std::string file);
    static void write_base64_mesh(const Mesh& mesh, uint cell_dim,
                                  std::string file, bool compress);

    // Compute base64 encoded stream for VTK
    template<typename T>
    static std::string encode_inline_base64(const std::vector<T>& data);

    // Compute compressed base64 encoded stream for VTK
    template<typename T>
    static std::string encode_inline_compressed_base64(const std::vector<T>& data);

  };

}

#endif
