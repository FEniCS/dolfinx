// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_CONVDIFF2D_cG1cG1_HH
#define __EQUATION_CONVDIFF2D_cG1cG1_HH

#include <Equation.hh>

class EquationConvDiff2d_cG1cG1: public Equation{
public:
  
  EquationConvDiff2d_cG1cG1():Equation(3){

	 AllocateFields(5);

	 field[0] = &beta[0];
	 field[1] = &beta[1];
	 field[2] = &up;
	 field[3] = &eps;
	 field[4] = &source;
	 
  } 

  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){

    MASS = u*v;
    LAP  = 0.5 * (u.dx*v.dx + u.dy*v.dy);
    CON  = 0.5 * (u.dx*(beta[0]*v) + u.dy*(beta[1]*v));

    return ( MASS + dt*(LAP+CON) );
  }
  
  real IntegrateRHS(ShapeFunction &v){
    
    MASS   = up*v;
    LAP    = 0.5 * (up.dx*v.dx + up.dy*v.dy);
    CON    = 0.5 * (up.dx*(beta[0]*v) + up.dy*(beta[1]*v));
    SOURCE = source*v;

    return ( MASS + dt*(SOURCE-LAP-CON) );
  }
  

private:
  
  LocalField up;
  LocalField beta[2];
  LocalField eps;
  LocalField source;
  
  real MASS,LAP,CON,SOURCE;

};

#endif
