// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_BLOCK_H
#define __ELEMENT_BLOCK_H

#include <dolfin/NewArray.h>
#include <dolfin/Component.h>
#include <dolfin/Element.h>

namespace dolfin {

  /// ElementBlock is a container for Elements, which are organized
  /// in a list of Components, each containing a list of Elements.

  class ElementBlock {
  public:

    /// Constructor
    ElementBlock(int N);

    /// Destructor
    ~ElementBlock();

    /// Create a new element
    Element* createElement(Element::Type type, unsigned int q, unsigned int index,
			   real t0, real t1);

    /// Return element for given component at given time
    Element* element(unsigned int, real t);
    
    /// Return first element for given component
    Element* first(unsigned int i);

    /// Return last element for given component
    Element* last(unsigned int i);

    /// Return number of components
    unsigned int size() const;

    /// Return number of bytes (approximately) used by this block
    unsigned int bytes() const;

    /// Return start time of block
    real starttime() const;

    /// Return end time of block
    real endtime() const;

    /// Return distance to given interval
    real dist(real t0, real t1) const;

    /// Check if given time is within this block
    bool within(real t) const;

    friend class ElementTmpFile;

  private:

    // Update interval
    void update(real t0, real t1);

    // List of components
    NewArray<Component> components;

    // Interval
    real t0, t1;

    // True if this block is empty
    bool empty;

    // Number of bytes (approximately) used by this block
    unsigned int _bytes;

  };

}

#endif
