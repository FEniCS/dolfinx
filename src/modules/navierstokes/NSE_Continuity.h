// Copyright (C) 2004 Johan Hoffman.
// Licensed under the GNU GPL Version 2.

#ifndef __NSE_CONTINUITY_H
#define __NSE_CONTINUITY_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class NSE_Continuity : public PDE {
  public:
    
    NSE_Continuity(Function::Vector& source,
		   Function::Vector& convection) : PDE(1, 1)
      {
	add(f,  source);
	add(b,  convection);

	C1 = 2.0;
      }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(up(0))+sqr(up(1))+sqr(up(2)));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);
      
      return
	(  d1*(grad(u),grad(v)) ) * dK

    }
    
    real rhs(ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(up(0))+sqr(up(1))+sqr(up(2)));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      return
	( - d1*((b,grad(up(0)))*v.dx() + (b,grad(up(1)))*v.dy() + (b,grad(up(2)))*v.dz()) - 
	  (b.dx() + b.dy() + b.dz())*v ) * dK

    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector b;   // Convection = linearized velocity

    real d1,d2,C1,C2,unorm;
  };
  
}

#endif
