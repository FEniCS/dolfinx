// Copyright (C) 2004 Johan Hoffman.
// Licensed under the GNU GPL Version 2.

#ifndef __NSE_H
#define __NSE_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class NSE_Momentum : public PDE {
  public:
    
    NSE_Momentum(Function::Vector& source,
		 Function::Vector& uprevious,
		 Function::Vector& convection,
		 Function& pressure) : PDE(3, 3)
      {
	add(f,  source);
	add(up, uprevious);
	add(b,  convection);
	add(p,  pressure);

	C1 = 2.0;
	C2 = 1.0;

	nu = 1.0/1000.0;
      }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(up(0)(cell->node(0).id()))+sqr(up(1)(cell->node(0).id()))+sqr(up(2)(cell->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      if ( (h/nu) > 1.0 ) d2 = C2 * h;
      else d2 = C2 * sqr(h);
      
      return
	( (u(0)*v(0) + u(1)*v(1) + u(2)*v(2))*(1.0/k) + 0.5 * 
	  (nu*((grad(u(0)),grad(v(0))) + (grad(u(1)),grad(v(1))) + (grad(u(2)),grad(v(2)))) + 
	   (b,grad(u(0)))*v(0) + (b,grad(u(1)))*v(1) + (b,grad(u(2)))*v(2) + 
	   d1*((b,grad(u(0)))*(b,grad(v(0))) + (b,grad(u(1)))*(b,grad(v(1))) + (b,grad(u(2)))*(b,grad(v(2)))) + 
	   d2*(u(0).dx()+u(1).dy()+u(2).dz())*(v(0).dx()+v(1).dy()+v(2).dz())) ) * dK;

    }
    
    real rhs(ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(up(0)(cell->node(0).id()))+sqr(up(1)(cell->node(0).id()))+sqr(up(2)(cell->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      if ( (h/nu) > 1.0 ) d2 = C2 * h;
      else d2 = C2 * sqr(h);
      
      return
	( (up(0),v(0) + up(1),v(1) + up(2),v(2))*(1.0/k) - 0.5 * 
	  (nu*( dx(up(0))*v(0).dx() + dy(up(0))*v(0).dy() + dz(up(0))*v(0).dz() + 
	        dx(up(1))*v(1).dx() + dy(up(1))*v(1).dy() + dz(up(1))*v(1).dz() + 
	        dx(up(2))*v(2).dx() + dy(up(2))*v(2).dy() + dz(up(2))*v(2).dz() ) + 
	   (b,grad(up(0)))*v(0) + (b,grad(up(1)))*v(1) + (b,grad(up(2)))*v(2) + 
	   d1*((b,grad(up(0)))*(b,grad(v(0))) + (b,grad(up(1)))*(b,grad(v(1))) + (b,grad(up(2)))*(b,grad(v(2)))) +   
	   d2*(up(0).dx()+up(1).dy()+up(2).dz())*(v(0).dx()+v(1).dy()+v(2).dz())) -
	  d1*(p.dx()*(b,grad(v(0))) + p.dy()*(b,grad(v(1))) + p.dz()*(b,grad(v(2)))) + 
	  p*(v(0).dx()+v(1).dy()+v(2).dz()) ) * dK;

    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector up;  // Velocity value at left end-point
    ElementFunction::Vector b;   // Convection = linearized velocity
    ElementFunction p;           // linearized pressure

    real nu,d1,d2,C1,C2,unorm;
  };

  //-------------------------------------------------------------------------------------------

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

      return
	( (grad(u),grad(v)) ) * dK;

    }
    
    real rhs(ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(up(0)(cell->node(0).id()))+sqr(up(1)(cell->node(0).id()))+sqr(up(2)(cell->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);
      
      return
	( - ((b,grad(up(0)))*v.dx() + (b,grad(up(1)))*v.dy() + (b,grad(up(2)))*v.dz()) - 
	  (1.0/d1) * (b.dx() + b.dy() + b.dz())*v ) * dK;

    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector b;   // Convection = linearized velocity

    real d1,d2,C1,C2,unorm;
  };
  


  
}

#endif
