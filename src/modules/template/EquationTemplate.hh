// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_TEMPLATE_HH
#define __EQUATION_TEMPLATE_HH

#include <Equation.hh>

class EquationTemplate: public Equation{
public:

  EquationTemplate():Equation(3){

	// Initialize local fields if any
	// AllocateFields(2);
	// field[0] = &f;
	// field[1] = &g;
	// ...
	 
  }
  
  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){
	 return ( u.dx*v.dx + u.dy*v.dy + u.dz*v.dz );
  }
  real IntegrateRHS(ShapeFunction &v){
	 return ( f * v );
  }

private:

  LocalField f,g;
  
};

#endif
