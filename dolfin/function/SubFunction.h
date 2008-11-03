// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-27
// Last changed: 2008-11-03

#ifndef __SUB_FUNCTION_H
#define __SUB_FUNCTION_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Function;

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

  class SubFunction
  {
  public:

    /// Create sub function for given component
    SubFunction(const Function& v, uint i) : v(v)
    {
      component.push_back(i);
    }

    /// Destructor
    ~SubFunction() {}

    /// Friends
    friend class Function;

  private:

    // The function
    const Function& v;

    // The component
    std::vector<uint> component;

  };

}

#endif
