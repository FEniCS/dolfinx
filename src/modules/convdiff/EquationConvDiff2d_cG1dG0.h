// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_CONVDIFF2D_cG1dG0_H
#define __EQUATION_CONVDIFF2D_cG1dG0_H

#include <Equation.h>

class EquationConvDiff2d_cG1dG0: public Equation{
public:
  
  EquationConvDiff2d_cG1dG0():Equation(2){

	 AllocateFields(5);
	 
	 field[0] = &up;
	 field[1] = &eps;
	 field[2] = &source;
	 field[3] = &beta[0];
	 field[4] = &beta[1];
	 
  } 

  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){

    MASS = u*v;
    LAP  = eps*(u.dx*v.dx + u.dy*v.dy);
    CONV = u.dx*(beta[0]*v) + u.dy*(beta[1]*v);

    return ( MASS + dt*(LAP+CONV) );
  }
  
  real IntegrateRHS(ShapeFunction &v){
    
    MASS   = up*v;
    SOURCE = source*v;

    return ( MASS + dt*SOURCE );
  }
  

private:
  
  LocalField up;
  LocalField eps;
  LocalField source;
  LocalField beta[2];
  
  real MASS,LAP,CONV,SOURCE;

};

#endif
