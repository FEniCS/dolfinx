// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_ELEMENT_H
#define __CGQ_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>

namespace dolfin {

  class RHS;

  class cGqElement : public Element {
  public:
    
    cGqElement(real t0, real t1, unsigned int q, unsigned int index);
    ~cGqElement();

    real value(real t) const;
    real dx() const;

    void update(real u0);
    void update(RHS& f);
    
    real computeTimeStep(real TOL, real r, real kmax) const;
    
  private:
    
    void feval(RHS& f);
    real integral(unsigned int i) const;
    
  };   
    
}

#endif
