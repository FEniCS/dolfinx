// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLUTION_H
#define __SOLUTION_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>
#include <dolfin/NewArray.h>

namespace dolfin {

  class ODE;
  class RHS;
  class ElementData;

  /// Solution represents the solution currently being computed.
  /// It contains element data and current (latest) initial value.
  /// In many ways a Solution is similar to a Function (ODEFunction),
  /// but differs in a number of ways:
  ///
  /// - It contains extra data (initial values) used to
  ///   propagate the solution.
  /// - It can do extrapolation.
  /// - It can create new elements.

  class Solution
  {
  public:
    
    /// Constructor
    Solution(ODE& ode, ElementData& elmdata);

    /// Destructor
    ~Solution();

    /// Create a new element
    Element* createElement(Element::Type type, unsigned int q, unsigned int index, real t0, real t1);

    /// Return element for given component at given time (null if not found)
    Element* element(unsigned int i, real t);

    /// Return first element element for given component (null if no elements)
    Element* first(unsigned int i);

    /// Return last element for given component (null if no elements)
    Element* last(unsigned int i);

    /// Evaluation (same as function u())
    real operator() (unsigned int i, real t);

    /// Return value for given component at given time
    real u(unsigned int i, real t);

    /// Return time step at given time
    real k(unsigned int i, real t);

    /// Return residual at given time
    real r(unsigned int i, real t, RHS& f);

    /// Return number of components
    unsigned int size() const;

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
