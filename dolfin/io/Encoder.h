// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-11
// Last changed: 

#ifndef __ENCODER_H
#define __ENCODER_H

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>

/// This class provides tools for encoding an compressing streams for use in 
/// output files

namespace dolfin
{

  namespace Encoder
  {
    typedef 
      boost::archive::iterators::base64_from_binary<    // convert binary values ot base64 characters
        boost::archive::iterators::transform_width<   // retrieve 6 bit integers from a sequence of 8 bit bytes
          const char*, 6, 8> > 
     base64_text; // compose all the above operations in to a new iterator

    
      template<typename T>
      void encode_base64(const T* data, uint length, std::stringstream& encoded_data)
      {
        std::copy(base64_text(&data[0]), base64_text(&data[length]), std::ostream_iterator<char>(encoded_data));
      }

      template<typename T>
      void encode_base64(const T& data, std::stringstream& encoded_data)
      {
        // We are cheating here and relying on the vector data being contiguous 
        // in memory. This will be part of the upcoming C++ standard
        std::copy(base64_text(&data[0]), base64_text(&data[data.size()]), std::ostream_iterator<char>(encoded_data));
      }

  }
}

#endif
