// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_ELEMENT_H
#define __GENERIC_ELEMENT_H

#include <dolfin/constants.h>

namespace dolfin {

  class GenericElement {
  public:
    
    GenericElement(int q);
    ~GenericElement();

    virtual void update() = 0;

  protected:

    // Nodal values
    real* values;

    // Order
    int q;

  };   
    
}

#endif
