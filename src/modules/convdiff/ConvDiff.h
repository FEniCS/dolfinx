// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CONV_DIFF_H
#define __CONV_DIFF_H

#include <dolfin/Equation.h>

namespace dolfin {
  
  class ConvDiff : public Equation {
  public:
    
    ConvDiff(Function& source,
	     Function& previous,
	     Function& diffusion,
	     Function::Vector& convection) : Equation(3)
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
      return (up*v + k*f*v) * dK;
    }
    
    
  private:    
    ElementFunction f;         // Source term
    ElementFunction up;        // Function value at left end-point
    ElementFunction a;         // Diffusivity
    ElementFunction::Vector b; // Convection
  };

}

#endif
