// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __VECTOR_EXPRESSION_FUNCTION_H
#define __VECTOR_EXPRESSION_FUNCTION_H

#include <dolfin/VFunctionPointer.h>
#include <dolfin/ExpressionFunction.h>

namespace dolfin {

  class Cell;
  class FiniteElement;
  class NewFiniteElement;

  class VectorExpressionFunction : public ExpressionFunction {
  public:
    
    VectorExpressionFunction(vfunction f, int dim, int size);
    ~VectorExpressionFunction();
    
    // Evaluation of function
    real operator() (const Node&  n, real t) const;
    real operator() (const Node&  n, real t);
    real operator() (const Point& p, real t) const;
    real operator() (const Point& p, real t);
    real operator() (real x, real y, real z, real t) const;
    real operator() (real x, real y, real z, real t);

    // Update values of element function
    void update(FunctionSpace::ElementFunction &v,
		const FiniteElement& element,
		const Cell& cell, real t) const;

    // FIXME: works only for nodal basis
    void update(NewArray<real>& w, const Cell& cell, const NewFiniteElement& element) const;

  private:
    
    vfunction f;
   
    unsigned int dim;
    unsigned int size;
    
  };

}

#endif
