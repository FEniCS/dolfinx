// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_ELEMENT_H
#define __CGQ_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>

namespace dolfin {

  class RHS;
  class TimeSlab;

  class cGqElement : public Element {
  public:
    
    cGqElement(int q, int index, TimeSlab* timeslab);

    real eval(real t) const;
    real dx() const;

    void update(real u0);
    void update(RHS& f);
    
  private:
    
    void feval(RHS& f);
    real integral(int i) const;
    real computeTimeStep() const;
    
  };   
    
}

#endif
