// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DGQ_ELEMENT_H
#define __DGQ_ELEMENT_H

#include <dolfin/GenericElement.h>

namespace dolfin {

  class dGqElement : public GenericElement {
  public:
    
    dGqElement(int q) : GenericElement(q) {};

    void update();
    
  };   
    
}

#endif
