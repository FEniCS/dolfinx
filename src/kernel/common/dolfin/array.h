// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-08-20
// Last changed: 2007-08-20

#ifndef __MINIMAL_ARRAY_H
#define __MINIMAL_ARRAY_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// The array class is a minimal wrapper for a real-valued array
  /// that knows its own size. It is not yet another array class;
  /// it is only intended to be used to pass data through the SWIG
  /// generated interface. Note that users of this class are always
  /// responsible for allocating and deallocating data.

  class array
  {
  public:

    /// Constructor
    array(uint size, real* data) : size(size), data(data) {}

    /// Destructor
    array() {}

    /// Member access
    real& operator[] (uint i) { return data[i]; }

    /// Member access (const)
    const real& operator[] (uint i) const { return data[i]; }

    /// Size of the array
    uint size;
    
    /// Array data
    real* data;

  };

}

#endif
