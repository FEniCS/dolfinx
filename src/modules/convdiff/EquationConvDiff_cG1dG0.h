// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_CONVDIFF_cG1dG0_H
#define __EQUATION_CONVDIFF_cG1dG0_H

#include <Equation.h>

class EquationConvDiff_cG1dG0: public Equation{
public:
  
  EquationConvDiff_cG1dG0():Equation(3){

	 AllocateFields(6);

	 field[0] = &beta[0];
	 field[1] = &beta[1];
	 field[2] = &beta[2];
	 field[3] = &up;
	 field[4] = &eps;
	 field[5] = &source;
	 
  } 

  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){

    MASS = u*v;
    LAP  = eps*(u.dx*v.dx + u.dy*v.dy + u.dz*v.dz);
    CONV = u.dx*(beta[0]*v) + u.dy*(beta[1]*v) + u.dz*(beta[2]*v);

    return ( MASS + dt*(LAP+CONV) );
  }
  
  real IntegrateRHS(ShapeFunction &v){
    
    MASS   = up*v;
    SOURCE = source*v;

    return ( MASS + dt*SOURCE );
  }
  

private:
  
  LocalField up;
  LocalField beta[3];
  LocalField eps;
  LocalField source;
  
  real MASS,LAP,CONV,SOURCE;

};

#endif
