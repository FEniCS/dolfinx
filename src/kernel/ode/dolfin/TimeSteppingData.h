// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_STEPPING_DATA_H
#define __TIME_STEPPING_DATA_H

#include <fstream>
#include <dolfin/Element.h>
#include <dolfin/Component.h>
#include <dolfin/NewArray.h>
#include <dolfin/Regulator.h>

namespace dolfin {

  class ODE;
  class ElementData;

  /// TimeSteppingData contains data for adaptive time-stepping,
  /// that can be shared by different time slabs.

  class TimeSteppingData {
  public:

    /// Constructor
    TimeSteppingData(ODE& ode, ElementData& elmdata);

    /// Destructor
    ~TimeSteppingData();

    /// Create a new element
    Element* createElement(Element::Type type, unsigned int q, unsigned int index,
			   real t0, real t1);

    /// Return element for given component at given time (null if not found)
    Element* element(unsigned int i, real t);
    
    /// Return number of components
    unsigned int size() const;
    
    /// Return time step regulator for given component
    Regulator& regulator(unsigned int i);

    /// Return time step regulator for given component
    const Regulator& regulator(unsigned int i) const;

    /// Return value for given component at given time
    real u(unsigned int i, real t) const;

    /// Return time step at given time
    real k(unsigned int i, real t) const;

    /// Return residual at given time
    real r(unsigned int i, real t, RHS& f) const;
    
    /// Return tolerance
    real tolerance() const;

    /// Return maximum time step
    real maxstep() const;

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Update initial value for given component
    real updateu0(unsigned int i, real t, real u0);

    /// Prepare for next time slab
    void shift(TimeSlab& timeslab, RHS& f);

    /// Save debug info
    enum Action { create = 0, update };
    void debug(Element& element, Action action);

  private:

    // Element data
    ElementData& elmdata;

    // Regulators, one for each component
    NewArray<Regulator> regulators;

    // Initial values, one for each component
    NewArray<real> initval;

    // Time where current initial values are specified
    real t0;

    // Tolerance
    real TOL;

    // Maximum allowed time step
    real kmax;
    
    // Threshold for reaching end of interval
    real interval_threshold;

    // Save debug info to file 'timesteps.debug'
    bool _debug;
    std::ofstream file;

  };

}

#endif
