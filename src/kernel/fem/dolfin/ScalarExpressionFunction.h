// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SCALAR_EXPRESSION_FUNCTION_H
#define __SCALAR_EXPRESSION_FUNCTION_H

#include <dolfin/function.h>
#include <dolfin/ExpressionFunction.h>

namespace dolfin {

  class Cell;
  class FiniteElement;

  class ScalarExpressionFunction : public ExpressionFunction {
  public:
    
    ScalarExpressionFunction(function f);
    
    // Update values of element function
    void update(FunctionSpace::ElementFunction& v,
		const FiniteElement& element,
		const Cell& cell,
		real t) const;
    
    // Evaluation of function
    real operator() (real x, real y, real z, real t) const;
    real operator() (const Node&  n, real t) const;
    real operator() (const Point& p, real t) const;
    
  private:
    
    function f;
    
  };

}

#endif
