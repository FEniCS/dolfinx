// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __COMPONENT_H
#define __COMPONENT_H

#include <vector>
#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Element.h>

namespace dolfin {

  class Element;
  class TimeSlab;
  class RHS;

  class Component {
  public:
    
    /// Constructor
    Component();

    /// Destructor
    ~Component();

    /// Create a new element
    Element* createElement(Element::Type type, real t0, real t1, int q, int index);

    /// Return element at given time
    Element* element(real t);

    /// Return last element
    Element* last();

    /// Return number of elements in component
    unsigned int size() const;

  private:

    // Find element for given time
    Element* findpos(real t);

    // Comparison operator for location of element
    struct Less : public std::binary_function<Element*, Element* , bool>
    {      
      Less(Element *dummy, real t);
      bool operator()(const Element *x, const Element *y);

      Element *dummy;
      real t;
    };

    //--- Component data ---

    // A list of elements for this component
    std::vector<Element*> elements;

  };

}

#endif
