// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_ELEMENT_H
#define __CGQ_ELEMENT_H

#include <dolfin/GenericElement.h>

namespace dolfin {

  class cGqElement : public GenericElement {
  public:
    
    cGqElement(int q) : GenericElement(q) {};

    void update();
    
  };   
    
}

#endif
