// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EXPRESSION_FUNCTION_H
#define __EXPRESSION_FUNCTION_H

#include <dolfin/ElementFunction.h>
#include <dolfin/GenericFunction.h>

namespace dolfin {

  class Cell;
  class FiniteElement;
  
  class ExpressionFunction : public GenericFunction {
  public:
    
    ExpressionFunction(int dim, int size);

    // Update values of element function
    virtual void update(FunctionSpace::ElementFunction& v,
			const FiniteElement& element,
			const Cell& cell,
			real t) const = 0;
    
    // Evaluation of function
    virtual real operator() (real x, real y, real z, real t) const = 0;
    virtual real operator() (const Node&  n, real t) const = 0;
    virtual real operator() (const Point& p, real t) const = 0;
    
  };

}

#endif
