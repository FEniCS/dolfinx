// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed:

#ifndef __NO_DELETER_H
#define __NO_DELETER_H

namespace dolfin
{

  /// NoDeleter is a customised deleter intended for use with smart pointers.

  template <class T>
  class NoDeleter
  {
    public:
      void operator() (T *p) {}
  };

}

#endif
