// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_POISSON_HH
#define __EQUATION_POISSON_HH

#include <Equation.hh>

class EquationPoisson: public Equation{
public:

  EquationPoisson():Equation(3){

	 AllocateFields(1);
	 field[0] = &f;
	 
  }
  
  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){
	 return ( u.dx*v.dx + u.dy*v.dy + u.dz*v.dz );
  }
  real IntegrateRHS(ShapeFunction &v){
	 return ( f * v );
  }

private:

  LocalField f;
  
};

#endif
