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

  class ElementData;

  /// TimeSteppingData contains data for adaptive time-stepping,
  /// that can be shared by different time slabs.

  class TimeSteppingData {
  public:

    /// Constructor
    TimeSteppingData(ElementData& elmdata);

    /// Destructor
    ~TimeSteppingData();
    
    /// Return number of components
    unsigned int size() const;
    /// Return given component
    Component& component(unsigned int i);

    /// Return given component
    const Component& component(unsigned int i) const;

    /// Return time step regulator for given component
    Regulator& regulator(unsigned int i);

    /// Return time step regulator for given component
    const Regulator& regulator(unsigned int i) const;

    /// Return tolerance
    real tolerance() const;

    /// Return maximum time step
    real maxstep() const;

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Create a new element
    Element* createElement(Element::Type type, int q, int index, TimeSlab* timeslab);

    /// Prepare for next time slab
    void shift(TimeSlab& timeslab, RHS& f);

    /// Save debug info
    enum Action { create = 0, update };
    void debug(Element& element, Action action);

  private:

    // Element data
    ElementData& elmdata;

    // List of regulators
    NewArray<Regulator> regulators;

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
