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
		 Function::Vector& pressure) : PDE(3, 3)
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

      unorm = sqrt(sqr(b(0)(cell->node(0).id()))+sqr(b(1)(cell->node(0).id()))+sqr(b(2)(cell->node(0).id())));
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

      unorm = sqrt(sqr(b(0)(cell->node(0).id()))+sqr(b(1)(cell->node(0).id()))+sqr(b(2)(cell->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      if ( (h/nu) > 1.0 ) d2 = C2 * h;
      else d2 = C2 * sqr(h);
      
      return
	( (up(0),v(0) + up(1),v(1) + up(2),v(2))*(1.0/k) - 0.5 * 
	  (nu*( dx(up(0))*v(0).dx() + dy(up(0))*v(0).dy() + dz(up(0))*v(0).dz() + 
	        dx(up(1))*v(1).dx() + dy(up(1))*v(1).dy() + dz(up(1))*v(1).dz() + 
	        dx(up(2))*v(2).dx() + dy(up(2))*v(2).dy() + dz(up(2))*v(2).dz() ) + 
	   (b(0)*dx(up(0)) + b(1)*dy(up(0)) + b(2)*dz(up(0)))*v(0) + 
	   (b(0)*dx(up(1)) + b(1)*dy(up(1)) + b(2)*dz(up(1)))*v(1) + 
	   (b(0)*dx(up(2)) + b(1)*dy(up(2)) + b(2)*dz(up(2)))*v(2) + 
	   d1*((b(0)*dx(up(0)) + b(1)*dy(up(0)) + b(2)*dz(up(0)))*(b,grad(v(0))) + 
	       (b(0)*dx(up(1)) + b(1)*dy(up(1)) + b(2)*dz(up(1)))*(b,grad(v(0))) + 
	       (b(0)*dx(up(2)) + b(1)*dy(up(2)) + b(2)*dz(up(2)))*(b,grad(v(0)))) + 
	   d2*(dx(up(0))+dy(up(1))+dz(up(2)))*(v(0).dx()+v(1).dy()+v(2).dz())) -
	  d1*(dx(p(0))*(b,grad(v(0))) + dy(p(0))*(b,grad(v(1))) + dz(p(0))*(b,grad(v(2)))) + 
	  p(0)*(v(0).dx()+v(1).dy()+v(2).dz()) ) * dK;

    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector up;  // Velocity value at left end-point
    ElementFunction::Vector b;   // Convection = linearized velocity
    ElementFunction::Vector p;           // linearized pressure

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
	nu = 1.0/1000.0;
      }
    
    real lhs(ShapeFunction& u, ShapeFunction& v)
    {

      return
	( (grad(u),grad(v)) ) * dK;

    }
    
    real rhs(ShapeFunction& v)
    {

      unorm = sqrt(sqr(b(0)(cell->node(0).id()))+sqr(b(1)(cell->node(0).id()))+sqr(b(2)(cell->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);
      
      return
	( (-1.0)*((b(0)*dx(b(0)) + b(1)*dy(b(0)) + b(2)*dz(b(0)))*v.dx() + 
		  (b(0)*dx(b(1)) + b(1)*dy(b(1)) + b(2)*dz(b(1)))*v.dy() + 
		  (b(0)*dx(b(2)) + b(1)*dy(b(2)) + b(2)*dz(b(2)))*v.dz()) -  
	  (1.0/d1)*(dx(b(0)) + dy(b(1)) + dz(b(2)))*v ) * dK;
      
    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector b;   // Convection = linearized velocity

    real nu,d1,d2,C1,C2,unorm;
  };
  


  
}

#endif
