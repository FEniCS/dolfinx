// Copyright (C) 2009 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2009-08-11
// Last changed: 2011-11-14

#ifndef __ENCODER_H
#define __ENCODER_H

#ifdef HAS_ZLIB
#include <zlib.h>
extern "C"
{
  int compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
}
#endif

#include <sstream>
#include <vector>
#include <utility>
#include "base64.h"

namespace dolfin
{

  /// This class provides tools for encoding and compressing streams
  /// for use in output files

  /// We cheating in some functions by relying on std::vector data being
  /// contiguous in memory. This will be part of the upcoming C++ standard.

  namespace Encoder
  {
    template<typename T>
    static void encode_base64(const T* data, std::size_t length,
                              std::stringstream& encoded_data)
    {
      encoded_data << base64_encode((const unsigned char*) &data[0],
                                    length*sizeof(T));
    }

    template<typename T>
    static void encode_base64(const std::vector<T>& data,
                              std::stringstream& encoded_data)
    {
      encoded_data << base64_encode((const unsigned char*) &data[0],
                                    data.size()*sizeof(T));
    }

    #ifdef HAS_ZLIB
    template<typename T>
    static std::vector<unsigned char> compress_data(const std::vector<T>& data)
    {
      // Compute length of uncompressed data
      const unsigned long uncompressed_size = data.size()*sizeof(T);

      // Compute maximum length of compressed data
      unsigned long compressed_size = (uncompressed_size + (((uncompressed_size)/1000)+1)+12);;

      // Allocate space for compressed data
      std::vector<unsigned char> compressed_data(compressed_size);

      // Compress data
      if (compress((Bytef*) compressed_data.data(), &compressed_size,
                   (const Bytef*) data.data(), uncompressed_size) != Z_OK)
      {
        dolfin_error("Encoder.h",
                     "compress data when writing file",
                     "Zlib error while compressing data");
      }

      // Return data
      return compressed_data;
    }
    #endif

  }
}

#endif
