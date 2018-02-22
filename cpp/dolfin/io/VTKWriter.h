// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Encoder.h"
#include <cstdint>
#include <string>
#include <vector>

namespace dolfin
{

class Function;
class Mesh;

/// Write VTK Mesh representation

class VTKWriter
{
public:
  /// Mesh writer
  static void write_mesh(const Mesh& mesh, std::size_t cell_dim,
                         std::string file, bool binary, bool compress);

  /// Cell data writer
  static void write_cell_data(const Function& u, std::string file, bool binary,
                              bool compress);

  /// Form (compressed) base64 encoded string for VTK
  template <typename T>
  static std::string encode_stream(const std::vector<T>& data, bool compress);
  // friend class VTKFile;

private:
  // Write cell data (ascii)
  static std::string ascii_cell_data(const Mesh& mesh,
                                     const std::vector<std::size_t>& offset,
                                     const std::vector<double>& values,
                                     std::size_t dim, std::size_t rank);

  // Write cell data (base64)
  static std::string base64_cell_data(const Mesh& mesh,
                                      const std::vector<std::size_t>& offset,
                                      const std::vector<double>& values,
                                      std::size_t dim, std::size_t rank,
                                      bool compress);

  // Mesh writer (ascii)
  static void write_ascii_mesh(const Mesh& mesh, std::size_t cell_dim,
                               std::string file);

  // Mesh writer (base64)
  static void write_base64_mesh(const Mesh& mesh, std::size_t cell_dim,
                                std::string file, bool compress);

  // Get VTK cell type
  static std::uint8_t vtk_cell_type(const Mesh& mesh, std::size_t cell_dim);

  // Compute base64 encoded stream for VTK
  template <typename T>
  static std::string encode_inline_base64(const std::vector<T>& data);

  // Compute compressed base64 encoded stream for VTK
  template <typename T>
  static std::string
  encode_inline_compressed_base64(const std::vector<T>& data);
};

//--------------------------------------------------------------------------
template <typename T>
std::string VTKWriter::encode_stream(const std::vector<T>& data, bool compress)
{
  std::stringstream stream;

  if (compress)
  {
#ifdef HAS_ZLIB
    return encode_inline_compressed_base64(data);
#else
    warning("zlib must be configured to enable compressed VTK output. Using "
            "uncompressed base64 encoding instead.");
    return encode_inline_base64(data);
#endif
  }
  else
    return encode_inline_base64(data);
}
//--------------------------------------------------------------------------
template <typename T>
std::string VTKWriter::encode_inline_base64(const std::vector<T>& data)
{
  std::stringstream stream;

  const std::uint32_t size = data.size() * sizeof(T);
  Encoder::encode_base64(&size, 1, stream);
  Encoder::encode_base64(data, stream);

  return stream.str();
}
//--------------------------------------------------------------------------
#ifdef HAS_ZLIB
template <typename T>
std::string
VTKWriter::encode_inline_compressed_base64(const std::vector<T>& data)
{
  std::stringstream stream;

  std::uint32_t header[4];
  header[0] = 1;
  header[1] = data.size() * sizeof(T);
  header[2] = 0;

  // Compress data
  std::vector<unsigned char> compressed_data = Encoder::compress_data(data);

  // Length of compressed data
  header[3] = compressed_data.size();

  // Encode header
  Encoder::encode_base64(&header[0], 4, stream);

  // Encode data
  Encoder::encode_base64(compressed_data.data(), compressed_data.size(),
                         stream);

  return stream.str();
}
#endif
//--------------------------------------------------------------------------
}


