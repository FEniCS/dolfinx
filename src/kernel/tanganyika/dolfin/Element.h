// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_H
#define __ELEMENT_H

namespace dolfin {

  class GenericElement;

  /// An Element is the basic building block of the time slabs used in
  /// the multi-adaptive time-stepping and represents the restriction of
  /// a component of the solution to a local interval.
  ///
  /// Element is a wrapper for a GenericElement which can be either a
  /// cGqElement or a dGqElement.

  class Element {
  public:
    
    Element(int q);
    ~Element();

  private:

    GenericElement* element;

  };   
    
}

#endif
