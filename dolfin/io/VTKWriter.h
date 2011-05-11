// Copyright (C) 2010 Garth N. Wells
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
// First added:  2010-07-19
// Last changed:

#ifndef __VTK_WRITER_H
#define __VTK_WRITER_H

#include <string>
#include <vector>
#include <boost/cstdint.hpp>
#include <dolfin/common/types.h>

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
  //friend class VTKFile;

  private:

    // Write cell data (ascii)
    static std::string ascii_cell_data(const Mesh& mesh,
                                       const std::vector<uint>& offset,
                                       const std::vector<double>& values,
                                       uint dim, uint rank);

    // Write cell data (base64)
    static std::string base64_cell_data(const Mesh& mesh,
                                        const std::vector<uint>& offset,
                                        const std::vector<double>& values,
                                        uint dim, uint rank, bool compress);

    // Mesh writer (ascii)
    static void write_ascii_mesh(const Mesh& mesh, uint cell_dim,
                                 std::string file);

    // Mesh writer (base64)
    static void write_base64_mesh(const Mesh& mesh, uint cell_dim,
                                  std::string file, bool compress);

    // Get VTK cell type
    static boost::uint8_t vtk_cell_type(const Mesh& mesh, uint cell_dim);

    // Compute base64 encoded stream for VTK
    template<typename T>
    static std::string encode_inline_base64(const std::vector<T>& data);

    // Compute compressed base64 encoded stream for VTK
    template<typename T>
    static std::string encode_inline_compressed_base64(const std::vector<T>& data);

  };

}

#endif
