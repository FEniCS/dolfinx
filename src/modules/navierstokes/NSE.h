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
		 Function::Vector& pressure) :
      PDE(3, 3), f(1), up(3), b(3), p(1)
      {
	add(f,  source);
	add(up, uprevious);
	add(b,  convection);
	add(p,  pressure);

	C1 = 2.0;
	C2 = 1.0;

	nu = 1.0/1000.0;

	bcwt = 1.0e7;
      }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(b(0)(cell_->node(0).id()))+sqr(b(1)(cell_->node(0).id()))+sqr(b(2)(cell_->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      if ( (h/nu) > 1.0 ) d2 = C2 * h;
      else d2 = C2 * sqr(h);
      
      MASS = (u(0)*v(0) + u(1)*v(1) + u(2)*v(2)) * dx;
      STIFF = nu*((grad(u(0)),grad(v(0))) + (grad(u(1)),grad(v(1))) + (grad(u(2)),grad(v(2)))) * dx;
      NL = ((b,grad(u(0)))*v(0) + (b,grad(u(1)))*v(1) + (b,grad(u(2)))*v(2)) * dx; 

      SD = (d1*((b,grad(u(0)))*(b,grad(v(0))) + (b,grad(u(1)))*(b,grad(v(1))) + (b,grad(u(2)))*(b,grad(v(2)))) + 
	    d2*(ddx(u(0))+ddy(u(1))+ddz(u(2)))*(ddx(v(0))+ddy(v(1))+ddz(v(2)))) * dx;      

      BC = bcwt*(u(0)*v(0) + u(1)*v(1) + u(2)*v(2)) * ds;

      return (MASS*(1.0/k) + 0.5*(STIFF + NL + SD) + BC); 

    }
    
    real rhs(ShapeFunction::Vector& v)
    {

      unorm = sqrt(sqr(b(0)(cell_->node(0).id()))+sqr(b(1)(cell_->node(0).id()))+sqr(b(2)(cell_->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);

      if ( (h/nu) > 1.0 ) d2 = C2 * h;
      else d2 = C2 * sqr(h);
      
      MASS = ((up(0),v(0)) + (up(1),v(1)) + (up(2),v(2))) * dx;
      STIFF = nu*( ddx(up(0))*ddx(v(0)) + ddy(up(0))*ddy(v(0)) + ddz(up(0))*ddz(v(0)) +
		   ddx(up(1))*ddx(v(1)) + ddy(up(1))*ddy(v(1)) + ddz(up(1))*ddz(v(1)) + 
		   ddx(up(2))*ddx(v(2)) + ddy(up(2))*ddy(v(2)) + ddz(up(2))*ddz(v(2)) ) * dx; 
      NL = ( (b(0)*ddx(up(0)) + b(1)*ddy(up(0)) + b(2)*ddz(up(0)))*v(0) + 
	     (b(0)*ddx(up(1)) + b(1)*ddy(up(1)) + b(2)*ddz(up(1)))*v(1) + 
	     (b(0)*ddx(up(2)) + b(1)*ddy(up(2)) + b(2)*ddz(up(2)))*v(2) ) * dx;
      SD = ( d1*((b(0)*ddx(up(0)) + b(1)*ddy(up(0)) + b(2)*ddz(up(0)))*(b,grad(v(0))) + 
		 (b(0)*ddx(up(1)) + b(1)*ddy(up(1)) + b(2)*ddz(up(1)))*(b,grad(v(0))) + 
		 (b(0)*ddx(up(2)) + b(1)*ddy(up(2)) + b(2)*ddz(up(2)))*(b,grad(v(0)))) + 
	     d2*(ddx(up(0))+ddy(up(1))+ddz(up(2)))*(ddx(v(0))+ddy(v(1))+ddz(v(2))) ) * dx; 
      SDP = d1*(ddx(p(0))*(b,grad(v(0))) + ddy(p(0))*(b,grad(v(1))) + ddz(p(0))*(b,grad(v(2)))) * dx;
      SDF = d1*(f(0)*(b,grad(v(0))) + f(1)*(b,grad(v(1))) + f(2)*(b,grad(v(2)))) * dx;
      PDV = p(0)*(ddx(v(0))+ddy(v(1))+ddz(v(2))) * dx;
      F = ((f(0),v(0)) + (f(1),v(1)) + (f(2),v(2))) * dx;
      
      ud0 = 0.0; ud1 = 0.0; ud2 = 0.0;
      un0 = 0.0; un1 = 0.0; un2 = 0.0;
      BC = bcwt*((ud0-un0)*v(0) + (ud1-un1)*v(1) + (ud2-un2)*v(2)) * ds;
      
      return (MASS*(1.0/k) - 0.5*(STIFF + NL + SD) - SDP + SDF + PDV + F + BC);

    }
    
  private:    
    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector up;  // Velocity value at left end-point
    ElementFunction::Vector b;   // Convection = linearized velocity
    ElementFunction::Vector p;   // linearized pressure

    real nu,d1,d2,C1,C2,unorm,SD,SDP,SDF,MASS,STIFF,NL,PDV,F,BC,ud0,ud1,ud2,un0,un1,un2,bcwt;
  };

  //-------------------------------------------------------------------------------------------

  class NSE_Continuity : public PDE {
  public:
    
    NSE_Continuity(Function::Vector& source,
		   Function::Vector& convection) :
      PDE(1, 1), f(1), b(3)
      {
	add(f,  source);
	add(b,  convection);

	C1 = 2.0;

	nu = 1.0/1000.0;

	bcwt = 0.0;
      }
    
    real lhs(ShapeFunction& u, ShapeFunction& v)
    {

      STIFF = (grad(u),grad(v)) * dx;

      BC = bcwt*(u*v) * ds;

      return (STIFF + BC);

    }
    
    real rhs(ShapeFunction& v)
    {

      unorm = sqrt(sqr(b(0)(cell_->node(0).id()))+sqr(b(1)(cell_->node(0).id()))+sqr(b(2)(cell_->node(0).id())));
      if ( (h/nu) > 1.0 ) d1 = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(unorm/h) ));
      else d1 = C1 * sqr(h);
      
      SD = ((b(0)*ddx(b(0)) + b(1)*ddy(b(0)) + b(2)*ddz(b(0)))*v.ddx() + 
	    (b(0)*ddx(b(1)) + b(1)*ddy(b(1)) + b(2)*ddz(b(1)))*v.ddy() + 
	    (b(0)*ddx(b(2)) + b(1)*ddy(b(2)) + b(2)*ddz(b(2)))*v.ddz()) * dx;   
      
      DIV = (ddx(b(0)) + ddy(b(1)) + ddz(b(2)))*v * dx;

      ud = 0.0;
      un = 0.0;
      BC = bcwt*((ud-un)*v) * ds;

      return ( (-1.0)*(SD + (1.0/d1)*DIV) + BC);
      
    }
    
  private:

    ElementFunction::Vector f;   // Source term
    ElementFunction::Vector b;   // Convection = linearized velocity

    real nu,d1,d2,C1,C2,unorm,bcwt,STIFF,SD,DIV,BC,ud,un;

  };
  


  
}

#endif
