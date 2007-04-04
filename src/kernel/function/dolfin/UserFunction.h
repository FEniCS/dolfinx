// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-04

#ifndef __USER_FUNCTION_H
#define __USER_FUNCTION_H

#include <dolfin/GenericFunction.h>

namespace dolfin
{
  class Function;

  /// This class implements the functionality for a user-defined
  /// function defined by overloading the evaluation operator in
  /// the class Function.

  class UserFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Create user-defined function
    UserFunction(Function* f);

    /// Destructor
    ~UserFunction();

    /// Interpolate function on cell
    void interpolate(real coefficients[],
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(real* values,
                  const real* coordinates,
                  const ufc::cell& cell) const;

  private:

    // Pointer to Function with overloaded evaluation operator
    Function* f;

  };

}

#endif
