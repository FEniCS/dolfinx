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
// First added:  2009-08-11
// Last changed: 2009-08-12

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
#include <boost/shared_array.hpp>
#include <dolfin/common/types.h>
#include "base64.h"

namespace dolfin
{

  /// Thes functions class provide tools for encoding and compressing streams
  /// for use in output files

  /// We cheating in some functions by relying on std::vector data being
  /// contiguous in memory. This will be part of the upcoming C++ standard.

  namespace Encoder
  {

    template<typename T>
    static void encode_base64(const T* data, uint length,
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
    static std::pair<boost::shared_array<unsigned char>, dolfin::uint> compress_data(const std::vector<T>& data)
    {
      // Compute length of uncompressed data
      const unsigned long uncompressed_size = data.size()*sizeof(T);

      // Compute maxium length of compressed data
      unsigned long compressed_size = (uncompressed_size + (((uncompressed_size)/1000)+1)+12);;

      // Allocate space for compressed data
      boost::shared_array<unsigned char> compressed_data(new unsigned char[compressed_size]);

      // Compress data
      if (compress((Bytef*) compressed_data.get(), &compressed_size, (const Bytef*) &data[0], uncompressed_size) != Z_OK)
        error("Zlib error while compressing data.");

      // Make pair and return
      return std::make_pair(compressed_data, compressed_size);
    }
#endif

  }
}

#endif
