// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_DATA_H
#define __TIME_SLAB_DATA_H

#include <fstream>

#include <dolfin/Vector.h>
#include <dolfin/Array.h>
#include <dolfin/Table.h>
#include <dolfin/Element.h>
#include <dolfin/Component.h>

namespace dolfin {

  class ODE;
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

    /// Shift solution at endtime to new u0
    void shift(TimeSlab& timeslab);

    /// Save debug info
    enum Action { slab = 0, create, update };
    void debug(Element& element, Action action);

  private:
    
    // List of components
    std::vector<Component> components;

    // Initial residuals
    std::vector<real> residuals_initial;

    // Latest residuals
    std::vector<real> residuals_latest;

    // Save debug info to file 'timeslab.debug'
    bool _debug;
    std::ofstream file;
    
  };
}

#endif
