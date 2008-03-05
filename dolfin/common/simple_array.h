// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007
//
// First added:  2007-08-20
// Last changed: 2007-08-24

#ifndef __SIMPLE_ARRAY_H
#define __SIMPLE_ARRAY_H

#include <dolfin/main/constants.h>

namespace dolfin
{

  /// The array class is a minimal wrapper for an array
  /// that knows its own size. It is not yet another array class;
  /// it is only intended to be used to pass data through the SWIG
  /// generated interface. Note that users of this class are always
  /// responsible for allocating and deallocating data.

  template<class T>
  class simple_array
  {
  public:

    /// Constructor
    simple_array(uint size, T* data) : size(size), data(data) {}

    /// Destructor
    simple_array() {}

    /// Member access
    T& operator[] (uint i) { return data[i]; }

    /// Member access (const)
    const T& operator[] (uint i) const { return data[i]; }

    /// Size of the array
    uint size;
    
    /// Array data
    T* data;

  };

}

#endif
