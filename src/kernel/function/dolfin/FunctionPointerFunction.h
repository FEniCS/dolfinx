// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-05

#ifndef __FUNCTION_POINTER_FUNCTION_H
#define __FUNCTION_POINTER_FUNCTION_H

#include <dolfin/FunctionPointer.h>
#include <dolfin/GenericFunction.h>

namespace dolfin
{
  
  /// This class implements the functionality for a user-defined
  /// function defined by a function pointer.

  class FunctionPointerFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Create function from function pointer
    FunctionPointerFunction(FunctionPointer f);

    /// Destructor
    ~FunctionPointerFunction();

    /// Interpolate function on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(real* values,
                  const real* coordinates,
                  const ufc::cell& cell) const;

  private:
    
    // Function pointer to user-defined function
    FunctionPointer f;
    
  };

}

#endif
