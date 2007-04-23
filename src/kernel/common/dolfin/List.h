// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005.

#ifndef __LIST_H
#define __LIST_H

#include <list>
#include <iterator>

namespace dolfin
{

  /// List is a container that provides O(n) access time to elements
  /// and O(n) memory overhead. However, a List can be grown/shrunk without
  /// reallocation and spliced together with other lists, etc.
  ///
  /// It is a wrapper for std::list (doubly-linked list), so see the STL
  /// manual for further details: http://www.sgi.com/tech/stl/ 

  template <class T>
  class List : public std::list<T>
  {
  public:
    
    /// Create empty list
    List() : std::list<T>() {}

    /// Copy constructor
    List(const List<T>& x) : std::list<T>(x) {}
    
  };
  
}

#endif
