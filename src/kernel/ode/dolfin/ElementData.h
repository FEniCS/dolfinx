// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_DATA_H
#define __ELEMENT_DATA_H

#include <dolfin/NewList.h>
#include <dolfin/Element.h>

namespace dolfin {

  class ODE;
  class ElementBlock;

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
    ElementData(int N);

    /// Destructor
    ~ElementData();

    /// Create a new element
    Element* createElement(Element::Type type, real t0, real t1, int q, int index);
    
    /// Return element for given component at given time (null if not found)
    Element* element(unsigned int i, real t);

    /// Return last element for given component (null if not in memory)
    Element* last(unsigned int i);

    /// Clear all data
    void clear();
    
    /// Return number of components
    unsigned int size() const;

  private:

    // Find block for given time
    ElementBlock* findpos(real t);

    // Size of system;
    int N;
    
    // List of element blocks
    NewList<ElementBlock*> blocks;

    // Current block
    ElementBlock* current;

  };

}

#endif
