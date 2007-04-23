// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2003-2007.

#ifndef __ARRAY_H
#define __ARRAY_H

#include <dolfin/constants.h>
#include <vector>

namespace dolfin
{

  /// Array is a container that provides O(1) access time to elements
  /// and O(1) memory overhead.  
  ///
  /// It is a wrapper for std::vector, so see the STL manual for further
  /// details: http://www.sgi.com/tech/stl/
  
  template <class T>
  class Array : public std::vector<T>
  {
  public:

    /// Create empty array
    Array() : std::vector<T>() {}
    
    /// Create array of given size
    Array(uint n) : std::vector<T>(n) {}

    /// Copy constructor
    Array(const Array<T>& x) : std::vector<T>(x) {}

    /// Assign to all elements in the array
    const Array& operator=(const T& element)
    {
      for (uint i = 0; i < std::vector<T>::size(); i++)
      (*this)[i] = element;
      return *this;
    }

  };

}

#endif
