// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OLDPOISSON_H
#define __OLDPOISSON_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class OldPoisson : public PDE {
  public:
    
    OldPoisson(Function& source) : PDE(3)
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
