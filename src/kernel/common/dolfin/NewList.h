// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_LIST_H
#define __NEW_LIST_H

#include <list>
#include <iterator>

namespace dolfin {

  /// A NewList is a container that provides O(n) access time to elements
  /// and O(n) memory overhead. However, a NewList can be grown/shrunk without
  /// reallocation and spliced together with other lists, etc.
  ///
  /// It is a wrapper for std::list (doubly-linked list), so see the STL
  /// manual for further details.

  template <class T>
  class NewList : public std::list<T>
  {
  public:
    
    NewList() : std::list<T>() {}

    NewList(NewList<T> &x) : std::list<T>(x) {}
    
  };
  
}

#endif
