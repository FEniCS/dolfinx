// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_H
#define __ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Vector.h>

namespace dolfin {

  class RHS;
  class TimeSlab;

  /// An Element is the basic building block of the time slabs used in
  /// the multi-adaptive time-stepping and represents the restriction of
  /// a component of the solution to a local interval.

  class Element {
  public:
    
    enum Type {cg, dg};

    /// Constructor
    Element(int q, int index, TimeSlab* timeslab);

    /// Destructor
    virtual ~Element();
    
    /// Evaluate element at given time
    virtual real eval(real t) const = 0;

    /// Evaluate element at given node within element
    real eval(int node) const;

    /// Evaluate element at the end point
    real eval() const;

    /// Evaluate derivative at the end point
    virtual real dx() const = 0;

    /// Update initial value
    virtual void update(real u0) = 0;

    /// Update element (iteration)
    virtual void update(RHS& f) = 0;

    /// Check if given time is within the element
    int within(real t) const;

    /// Check if the element is within the given time slab
    bool within(TimeSlab* timeslab) const;

    /// Return the left end-point
    real starttime() const;

    /// Return the right end-point
    real endtime() const;

    /// Return the size of the time step
    real timestep() const;

    /// Component index
    int index;

    // Compute residual
    real computeResidual(RHS& f);

    // Compute new time step
    virtual real computeTimeStep() const = 0;

  protected:

    // Evaluate the right-hand side
    virtual void feval(RHS& f) = 0;

    // Compute integral for degree of freedom i using quadrature
    virtual real integral(int i) const = 0;

    // Temporary storage for function evaluations (common to all elements).
    static Vector f;

    // --- Element data ---
    
    // Nodal values
    real* values;

    // Order
    int q;
    
    // The time slab this element belongs to
    TimeSlab* timeslab;

  };   
    
}

#endif
