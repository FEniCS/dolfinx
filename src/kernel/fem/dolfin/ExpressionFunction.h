// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EXPRESSION_FUNCTION_H
#define __EXPRESSION_FUNCTION_H

#include <dolfin/function.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/GenericFunction.h>

namespace dolfin {

  class Cell;
  class FiniteElement;
  
  class ExpressionFunction : public GenericFunction {
  public:

	 ExpressionFunction(function f);
	 
	 // Update values of element function
	 void update(FunctionSpace::ElementFunction &v,
					 const FiniteElement &element,
					 const Cell &cell,
					 real t) const;

	 // Evaluation of function
	 real operator() (const Node&  n, real t) const;
	 real operator() (const Point& p, real t) const;

  private:

	 function f;
	 
  };

}

#endif
