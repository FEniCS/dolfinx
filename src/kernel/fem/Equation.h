// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_H
#define __EQUATION_H

#include <dolfin/Function.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/constants.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {

  class FiniteElement;
  
  class Equation {
  public:

	 Equation();
	 Equation(int nsd);
	 ~Equation();

	 // Allow simpler notation for classes defined in FunctionSpace
	 typedef FunctionSpace::ShapeFunction ShapeFunction;
	 typedef FunctionSpace::Product Product;
	 typedef FunctionSpace::ElementFunction ElementFunction;
  
	 virtual real lhs(ShapeFunction &u, ShapeFunction &v) = 0;
	 virtual real lhs(ShapeFunction &v) = 0;
	 
	 void updateLHS(FiniteElement *element);
	 void updateRHS(FiniteElement *element);
	 
	 void setTime     (real t);
	 void setTimeStep (real dt);

  protected:
	 
	 void updateCommon(FiniteElement *element);
	 
	 virtual void updateLHS() {};
	 virtual void updateRHS() {};
	 
	 int start_vector_component;
	 
	 int nsd;   // number of space dimensions
	 int no_eq; // number of equations
	 
	 real dt;
	 real t;  
	 real h;
	 
  };

}

#endif
