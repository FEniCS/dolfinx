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

    Element(int q, int index, TimeSlab* timeslab);
    virtual ~Element();
    
    virtual real eval(real t) const = 0;
    virtual real eval(int node) const = 0;

    virtual void update(real u0) = 0;
    virtual void update(RHS& f) = 0;

    int within(real t) const;
    bool within(TimeSlab* timeslab) const;

    real starttime() const;
    real endtime() const;
    real timestep() const;

    virtual real newTimeStep() const = 0;

    // Component index
    int index;

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
