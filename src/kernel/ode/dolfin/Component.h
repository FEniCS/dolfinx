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

    /// Constructor
    Component(int size);

    /// Destructor
    ~Component();

    /// Return element at given time
    Element& element(real t);

    /// Return last element
    Element& last();
    
    /// Return number of elements in component
    int size() const;

    /// Return value at given time
    real value(real t);

    /// Return time step at given time
    real timestep(real t);

    /// Return residual at given time
    real residual(real t, RHS& f);

    /// Evaluation operator
    real operator() (real t);
    
    /// Evaluation operator
    real operator() (int node, real t, TimeSlab* timeslab);

    /// Create a new element
    Element* createElement(Element::Type type,
			   int q, int index, TimeSlab* timeslab);

    /// Clear component
    void clear();
        
    friend class TimeSlabData;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const Component& data);

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

    // Initial value
    real u0;

  };

}

#endif
