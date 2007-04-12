// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2007-04-12

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <ufc.h>
#include <dolfin/constants.h>

namespace dolfin
{
  
  /// This class serves as a base class/interface for implementations
  /// of specific function representations.

  class GenericFunction
  {
  public:

    /// Constructor
    GenericFunction() {};

    /// Destructor
    virtual ~GenericFunction() {};

    /// Return the rank of the value space
    virtual uint rank() const = 0;

    /// Return the dimension of the value space for axis i
    virtual uint dim(uint i) const = 0;

    /// Interpolate function on cell
    virtual void interpolate(real* coefficients,
                             const ufc::cell& cell,
                             const ufc::finite_element& finite_element) = 0;

  };

}

#endif
