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
    
    cGqElement(unsigned int q, unsigned int index, real t0, real t1);
    ~cGqElement();

    Type type() const;

    real value(real t) const;
    real initval() const;
    real dx() const;

    void update(real u0);
    void update(RHS& f);
    void update(RHS& f, real alpha);

    void reset(real u0);

    real computeTimeStep(real TOL, real r, real kmax) const;
    real computeDiscreteResidual(RHS& f);

  private:
    
    void feval(RHS& f);
    real integral(unsigned int i) const;
    
  };   
    
}

#endif
