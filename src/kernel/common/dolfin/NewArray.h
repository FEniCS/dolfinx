// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Updates by Anders Logg, 2003, 2004

#ifndef __NEWARRAY_H
#define __NEWARRAY_H

#include <dolfin/constants.h>
#include <vector>

namespace dolfin {

  /// A NewArray is a container that provides O(1) access time to elements
  /// and O(1) memory overhead.  
  ///
  /// It is a wrapper for std::vector, so see the STL manual for further
  /// details.
  
  template <class T>
  class NewArray : public std::vector<T>
  {
  public:

    /// Constructor
    NewArray() : std::vector<T>() {}
    
    /// Constructor
    NewArray(uint n) : std::vector<T>(n) {}

    /// Assign to all elements in the array
    const NewArray& operator=(const T& element)
    {
      for (uint i = 0; i < std::vector<T>::size(); i++)
	(*this)[i] = element;
      return *this;
    }

  };

}

#endif
