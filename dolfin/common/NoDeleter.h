// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008.
//
// First added:  2008-09-11
// Last changed: 2008-12-04

#ifndef __NO_DELETER_H
#define __NO_DELETER_H

#include <boost/shared_ptr.hpp>

namespace dolfin
{

  /// NoDeleter is a customised deleter intended for use with smart pointers.

  template <class T>
  class NoDeleter
  {
  public:
      void operator() (T *p) {}
  };

  /// Helper function to construct shared pointer with NoDeleter with cleaner syntax

  template<class T>
  boost::shared_ptr<T> reference_to_no_delete_pointer(T& r)
  {
    return boost::shared_ptr<T>(&r, NoDeleter<T>());
  }

}

#endif
