// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_H
#define __ELEMENT_H

#include <dolfin/constants.h>

namespace dolfin {

  class RHS;
  class GenericElement;
  class TimeSlab;

  /// An Element is the basic building block of the time slabs used in
  /// the multi-adaptive time-stepping and represents the restriction of
  /// a component of the solution to a local interval.
  ///
  /// Element is a wrapper for a GenericElement which can be either a
  /// cGqElement or a dGqElement.

  class Element {
  public:

    enum Type {cg, dg};

    /// Constructor
    Element();

    /// Constructor
    Element(Type type, int q, int index, int pos, TimeSlab* timeslab);

    /// Destructor
    ~Element();
    
    /// Initialize element to given method and order
    void init(Type type, int q, int index, int pos, TimeSlab* timeslab);

    /// Evaluation at given time t
    real eval(real t) const;

    // Evaluation at given nodal point
    real eval(int node) const;

    /// Update initial value for element
    void update(real u0);

    /// Update element values (iteration)
    void update(RHS& f);
    
    /// Check if given discrete time is within the element
    int within(real t) const;

    /// Check if the element is within the given time slab
    bool within(TimeSlab* timeslab) const;

    /// Return start time
    real starttime() const;

    /// Return end time
    real endtime() const;

    /// Return time step
    real timestep() const;

    /// Compute new time step for next element
    real newTimeStep() const;

    /// Assignment
    void operator=(Element& element);

  private:

    GenericElement* element;

  };   
    
}

#endif
