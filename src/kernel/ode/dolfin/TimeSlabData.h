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
    
    /// Specify top level time slab
    void setslab(TimeSlab* timeslab);

    /// Return number of elements
    int size() const;

    /// Return given component
    Component& component(int i);

    /// Shift solution at endtime to new u0
    void shift();

  private:

    // List of components
    std::vector<Component> components;

    // Initial residuals
    std::vector<real> residuals_initial;

    // Latest residuals
    std::vector<real> residuals_latest;

    // Top level time slab
    TimeSlab* topslab;

    // Save debug info to file 'timeslab.debug'
    bool debug;
    std::ofstream file;
    
  };
}

#endif
