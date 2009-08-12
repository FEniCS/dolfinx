// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-11
// Last changed: 2009-08-12

#ifndef __ENCODER_H
#define __ENCODER_H

#include <vector>
#include "base64.h"

namespace dolfin
{

  /// This class provides tools for encoding an compressing streams for use in 
  /// output files

  namespace Encoder
  {
      template<typename T>
      void encode_base64(const T* data, uint length, std::stringstream& encoded_data)
      {
        encoded_data << base64_encode((const unsigned char*) &data[0], length*sizeof(T));
      }

      template<typename T>
      void encode_base64(const std::vector<T>& data, std::stringstream& encoded_data)
      {
        // We are cheating here and relying on the vector data being contiguous 
        // in memory. This will be part of the upcoming C++ standard
        encoded_data << base64_encode((const unsigned char*) &data[0], data.size()*sizeof(T));
      }
  }
}

#endif
