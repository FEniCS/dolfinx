// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DGQ_ELEMENT_H
#define __DGQ_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>

namespace dolfin {

  class RHS;
  class TimeSlab;

  class dGqElement : public Element {
  public:
    
    dGqElement(int q, int index, TimeSlab* timeslab);

    real eval(real t) const;
    real eval(int node) const;

    void update(real u0);
    void update(RHS& f);

    real newTimeStep() const;
    
  private:

    void feval(RHS& f);
    real integral(int i) const;

    // End value from previous interval
    real u0;

  };   
    
}

#endif
