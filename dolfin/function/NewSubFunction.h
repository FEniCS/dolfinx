// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-27
// Last changed: 2008-10-12

#ifndef __NEW_SUB_FUNCTION_H
#define __NEW_SUB_FUNCTION_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class NewFunction;

  /// This class represents a sub function (view) of a function.
  /// It's purpose is to enable expressions like
  ///
  ///    Function w;
  ///    Function u = w[0];
  ///    Function p = w[1];
  ///
  /// without needing to create and destroy temporaries. No data is created
  /// until a SubFunction is assigned to a Function, at which point the data
  /// is copied.

  class NewSubFunction
  {
  public:

    /// Create sub function
    NewSubFunction(const NewFunction& v, uint i) : v(v), i(i) {}

    /// Destructor
    ~NewSubFunction() {}

    /// Friends
    friend class NewFunction;

  private:

    // The function
    const NewFunction& v;

    // The sub function index
    uint i;

  };

}

#endif
