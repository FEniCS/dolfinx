// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_CONVDIFF_cG1cG1_HH
#define __EQUATION_CONVDIFF_cG1cG1_HH

#include <Equation.hh>

class EquationConvDiff_cG1cG1: public Equation{
public:

  // Warning: Not up-to-date
  
  EquationConvDiff_cG1cG1():Equation(3){

	 AllocateFields(6);

	 field[0] = &up;
	 field[1] = &eps;
	 field[2] = &source;
	 field[3] = &beta[0];
	 field[4] = &beta[1];
	 field[5] = &beta[2];
	 
  } 

  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){

    MASS = u*v;
    LAP  = 0.5 * (u.dx*v.dx + u.dy*v.dy + u.dz*v.dz);
    CON  = 0.5 * (u.dx*(beta[0]*v) + u.dy*(beta[1]*v) + u.dz*(beta[2]*v));

    return ( MASS + dt*(LAP+CON) );
  }
  
  real IntegrateRHS(ShapeFunction &v){
    
    MASS   = up*v;
    LAP    = 0.5 * (up.dx*v.dx + up.dy*v.dy + up.dz*v.dz);
    CON    = 0.5 * (up.dx*(beta[0]*v) + up.dy*(beta[1]*v) + up.dz*(beta[2]*v));
    SOURCE = source*v;

    return ( MASS + dt*(SOURCE-LAP-CON) );
  }
  

private:
  
  LocalField up;
  LocalField eps;
  LocalField source;
  LocalField beta[3];
  
  real MASS,LAP,CON,SOURCE;

};

#endif
