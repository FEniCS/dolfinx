// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_ELEMENT_H
#define __GENERIC_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Vector.h>

namespace dolfin {

  class RHS;
  class TimeSlab;

  class GenericElement {
  public:
    
    GenericElement(int q, int index, int pos, TimeSlab* timeslab);
    ~GenericElement();
    
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
    
    // Component index
    int index;

    // Position in component list of elements
    int pos;

    // The time slab this element belongs to
    TimeSlab* timeslab;

  };   
    
}

#endif
