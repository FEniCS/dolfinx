// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_DATA_H
#define __TIME_SLAB_DATA_H

#include <fstream>

#include <dolfin/NewArray.h>
#include <dolfin/Component.h>
#include <dolfin/Regulator.h>
#include <dolfin/Element.h>

namespace dolfin {

  class ODE;
  class RHS;
  class TimeSlab;

  /// TimeSlabData is a container for time slab data (elements).
  /// A block-linked list is used to store the elements.
  /// The purpose of this class is to be able to reuse elements
  /// from previous time slabs in new time slabs.
  
  class TimeSlabData {
  public:

    /// Constructor
    TimeSlabData(ODE& ode);

    /// Destructor
    ~TimeSlabData();

    /// Create element
    Element* createElement(Element::Type type, int q, int index,
			   TimeSlab* timeslab);
    
    /// Return number of components
    unsigned int size() const;

    /// Return given component
    Component& component(unsigned int i);

    /// Return given component
    const Component& component(unsigned int i) const;

    /// Return given regulator
    Regulator& regulator(unsigned int i);

    /// Return given regulator
    const Regulator& regulator(unsigned int i) const;

    /// Return tolerance
    real tolerance() const;

    /// Return maximum time step
    real maxstep() const;

    /// Return threshold for reaching end of interval
    real threshold() const;
    
    /// Prepare for next time slab
    void shift(TimeSlab& timeslab, RHS& f);

    /// Save debug info
    enum Action { create = 0, update };
    void debug(Element& element, Action action);

    /// Output
    friend LogStream& operator<<(LogStream& stream, const TimeSlabData& data);

  private:
    
    // List of components
    NewArray<Component> components;

    // List of regulators
    NewArray<Regulator> regulators;

    // Tolerance
    real TOL;

    // Maximum allowed time step
    real kmax;
    
    // Threshold for reaching end of interval
    real interval_threshold;

    // Save debug info to file 'timeslab.debug'
    bool _debug;
    std::ofstream file;

  };
}

#endif
