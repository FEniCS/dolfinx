// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLUTION_H
#define __SOLUTION_H

#include <fstream>
#include <dolfin/constants.h>
#include <dolfin/Element.h>
#include <dolfin/NewArray.h>
#include <dolfin/Variable.h>

namespace dolfin {

  class ODE;
  class RHS;
  class Function;
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
  /// - It can save debug info to a file.

  class Solution : public Variable
  {
  public:
    
    /// Constructor
    Solution(ODE& ode, Function& u);

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
    real operator() (unsigned int i, unsigned int node, real t);

    /// Return value for given component at given time
    real u(unsigned int i, real t);

    /// Return value for given component at given time (optimized version)
    real u(unsigned int i, unsigned int node, real t);

    /// Return time step at given time
    real k(unsigned int i, real t);

    /// Return residual at given time
    real r(unsigned int i, real t, RHS& f);

    /// Return number of components
    unsigned int size() const;

    /// Return method to use for given component
    Element::Type method(unsigned int i);

    /// Return order to use for given component
    unsigned int order(unsigned int i);

    /// Prepare for next time slab (propagate values)
    void shift(real t0);

    /// Reset current element block
    void reset();

    /// Save debug info
    enum Action { create = 0, update };
    void debug(Element& element, Action action);

  private:

    // The ODE
    ODE& ode;

    // Element data
    ElementData& elmdata;

    // Initial values (propagated values)
    NewArray<real> u0;

    // Time where current initial values are specified
    real t0;
    
    // Save debug info to file 'timesteps.debug'
    bool _debug;
    std::ofstream file;

  };

}

#endif
