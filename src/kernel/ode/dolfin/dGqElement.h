// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DGQ_ELEMENT_H
#define __DGQ_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>

namespace dolfin {

  class RHS;

  class dGqElement : public Element {
  public:
    
    dGqElement(unsigned int q, unsigned int index, real t0, real t1);
    ~dGqElement();

    Type type() const;

    unsigned int size() const;

    real value(real t) const;
    real value(unsigned int node, real t) const;
    real initval() const;
    real dx() const;

    void update(real u0);
    real update(RHS& f);
    real update(RHS& f, real alpha);
    real update(RHS& f, real alpha, real* values);

    void set(real u0);
    void set(const real* const values);
    void get(real* const values) const;

    bool accept(real TOL, real r);

    real computeTimeStep(real TOL, real r, real kmax) const;
    real computeDiscreteResidual(RHS& f);
    real computeElementResidual(RHS& f);
    
  private:

    void feval(RHS& f);
    real integral(unsigned int i) const;

    // End value from previous interval
    real u0;

  };   
    
}

#endif
