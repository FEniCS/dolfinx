// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_H
#define __ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Vector.h>

namespace dolfin {

  class RHS;

  /// An Element is the basic building block of the time slabs used in
  /// the multi-adaptive time-stepping and represents the restriction of
  /// a component of the solution to a local interval.

  class Element {
  public:
    
    enum Type {cg, dg};

    /// Constructor
    Element(unsigned int q, unsigned int index, real t0, real t1);

    /// Destructor
    virtual ~Element();

    /// Return type of element
    virtual Type type() const = 0;

    /// Return order of element
    unsigned int order() const;

    /// Return value of element at given time
    virtual real value(real t) const = 0;

    /// Return value of element at given node within element
    real value(unsigned int node) const;

    /// Return initial value of element
    virtual real initval() const = 0;

    /// Return value of element at the end point
    real endval() const;

    /// Return derivative at the end point
    virtual real dx() const = 0;

    /// Update initial value
    virtual void update(real u0) = 0;

    /// Update given value
    void update(unsigned int node, real value);

    /// Update element (iteration)
    virtual void update(RHS& f) = 0;

    /// Check if given time is within the element
    bool within(real t) const;

    /// Return component index
    unsigned int index() const;

    /// Return the left end-point
    real starttime() const;

    /// Return the right end-point
    real endtime() const;

    /// Return the size of the time step
    real timestep() const;

    // Compute residual
    real computeResidual(RHS& f);

    // Compute new time step
    virtual real computeTimeStep(real TOL, real r, real kmax) const = 0;

  protected:

    // Evaluate the right-hand side
    virtual void feval(RHS& f) = 0;

    // Compute integral for degree of freedom i using quadrature
    virtual real integral(unsigned int i) const = 0;

    // Temporary storage for function evaluations (common to all elements).
    static Vector f;

    //--- Element data ---
    
    // Order
    unsigned int q;

    /// Component index
    unsigned int _index;
    
    // Interval
    real t0, t1;

    // Nodal values
    real* values;
    
  };   
    
}

#endif
