// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __COMPONENT_H
#define __COMPONENT_H

#include <vector>
#include <functional>
#include <algorithm>
#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Element.h>

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

    /// Evaluation at given time
    real operator() (int node, real t, TimeSlab* timeslab);

    /// Evaluation at given time
    real operator() (real t);

    /// Create a new element
    Element* createElement(const Element::Type type,
			   int q, int index, TimeSlab* timeslab);

    /// Return last element
    Element& last();

    // Return number of elements in component
    int size();

    // Initial value
    real u0;

  private:

    // Find element for given discrete time
    Element *findpos(real t);

    // A list of elements for this component
    std::vector<Element*> elements;


    // Current position (latest position)
    int current;

    // Next available position
    int next;

    struct LessElement :
      public std::binary_function<Element *, Element *, bool>
    {
      Element *dummy;
      real t;
      
      LessElement(Element *dummy, real t);
      bool operator()(const Element *x, const Element *y);
    };

  };

}

#endif
