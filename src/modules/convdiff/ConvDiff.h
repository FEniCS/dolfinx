// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CONV_DIFF_H
#define __CONV_DIFF_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class ConvDiff : public PDE {
  public:
    
    ConvDiff(Function& source,
	     Function::Vector& previous,
	     Function& diffusion,
	     Function::Vector& convection) : PDE(2)
    {
      add(f,  source);
      add(up, previous);
      add(a,  diffusion);
      add(b,  convection);
    }
    
    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return (u*v + k*((b,grad(u))*v + a*(grad(u),grad(v)))) * dK;
    }
    
    real rhs(const ShapeFunction& v)
    {
      return (up(0)*v + k*f*v) * dK;
    }
    
  private:    
    ElementFunction f;         // Source term
    ElementFunction::Vector up;        // Function value at left end-point
    ElementFunction a;         // Diffusivity
    ElementFunction::Vector b; // Convection
  };

}

#endif
