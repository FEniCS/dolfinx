// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLUTION_H
#define __SOLUTION_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>

namespace dolfin {

  class ODE;
  class ElementData;

  /// Solution represents the solution currently being computed.
  /// It contains element data and current (latest) initial value.
  /// In many ways a Solution is similar to a Function (ODEFunction),
  /// but differs in two ways: it contains extra data (initial values)
  /// used to propagate the solution, and it can do extrapolation.

  class Solution
  {
  public:
    
    /// Constructor
    Solution(const ODE& ode, ElementData& elmdata);

    /// Destructor
    ~Solution();

    /// Evaluation
    real operator() (unsigned int i, real t);

    /// Prepare for next time slab (propagate values)
    void shift(real t0);

  private:

    // Element data
    ElementData& elmdata;

    // Initival values (propagated values)
    NewArray<real> u0;

    // Time where current initial values are specified
    real t0;

  };

}

#endif
