// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_DATA_H
#define __ELEMENT_DATA_H

#include <dolfin/NewArray.h>
#include <dolfin/Component.h>
#include <dolfin/Element.h>

namespace dolfin {

  class ODE;
  class RHS;
  class TimeSteppingData;
  class TimeSlab;

  /// ElementData is a container for Elements, which are organized
  /// in a sequence of ElementBlocks, each containing a list of
  /// Components, which each contain a list of Elements:
  ///
  ///     ElementData - ElementBlock - Component - Element
  ///
  /// A temporary file is used to store data whenever the amount of
  /// data exceeds the specified cache size.

  class ElementData {
  public:

    /// Constructor
    ElementData(const ODE& ode);

    /// Destructor
    ~ElementData();

    /// Create a new element
    Element* createElement(Element::Type type, int q, int index, TimeSlab* timeslab);
    
    /// Return number of components
    unsigned int size() const;

    /// Return given component
    Component& component(unsigned int i);

    /// Return given component
    const Component& component(unsigned int i) const;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const ElementData& data);

  private:
    
    // List of components
    NewArray<Component> components;

  };

}

#endif
