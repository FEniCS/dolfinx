// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_POISSON_2D_H
#define __EQUATION_POISSON_2D_H

#include <Equation.h>

class EquationPoisson2d: public Equation{
public:

  EquationPoisson2d():Equation(2){

	 AllocateFields(1);
	 field[0] = &f;
	 
  }
  
  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){
	 return ( u.dx*v.dx + u.dy*v.dy );
  }
  real IntegrateRHS(ShapeFunction &v){
	 return ( f * v );
  }

private:

  LocalField f;
  
};

#endif
