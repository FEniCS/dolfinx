// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_OLD_H
#define __POISSON_OLD_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class PoissonOld : public PDE {
  public:
    
    PoissonOld(Function& source) : PDE(3)
    {
      add(f, source);
    }
    
    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return (grad(u),grad(v))*dx;
    }
    
    real rhs(const ShapeFunction& v)
    {
      return f*v*dx;
    }
    
  private:
    
    ElementFunction f;
    
  };
  
}

#endif
