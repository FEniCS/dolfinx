// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HEAT_H
#define __HEAT_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class Heat : public PDE {
  public:
    
    Heat(Function& source,
	 Function& previous,
	 Function& diffusion) : PDE(3)
    {
      add(f,  source);
      add(up, previous);
      add(a,  diffusion);
    }
    
    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return (u*v + a*(grad(u),grad(v))) * dx;
    }
    
    real rhs(const ShapeFunction& v)
    {
      return (up*v + k*f*v) * dx;
    }
    
  private:    

    ElementFunction f;         // Source term
    ElementFunction up;        // Function value at left end-point
    ElementFunction a;         // Diffusivity

  };

}

#endif
