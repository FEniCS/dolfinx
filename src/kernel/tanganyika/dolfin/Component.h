// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __COMPONENT_H
#define __COMPONENT_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin {

  class Element;
  class TimeSlab;

  class Component {
  public:
    
    /// Constructor
    Component();

    /// Constructor
    Component(int size);

    /// Destructor
    ~Component();

    /// Initialize to given number of elements
    void init(int size);

    /// Evaluation at given time
    real operator() (int node, real t, TimeSlab* timeslab);

    /// Add a new element
    int add(Element& element, real t1);

    /// Return last element
    Element& last();

  private:

    // Find element for given discrete time
    Element findpos(real t);

    // A list of elements for this component
    Array<Element> elements;

    // Current position (latest position)
    int current;

    // Next available position
    int next;

  };

}

#endif
