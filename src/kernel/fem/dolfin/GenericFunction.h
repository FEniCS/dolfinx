// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <dolfin/ElementFunction.h>

namespace dolfin {

  class Cell;
  class Node;
  class Point;
  class FiniteElement;
  
  class GenericFunction {
  public:
	 
	 // Update values of element function
	 virtual void update(FunctionSpace::ElementFunction &v,
								const FiniteElement &element,
								const Cell &cell,
								real t) const = 0;

	 // Evaluation of function
	 virtual real operator() (const Node&  n, real t) const = 0;
	 virtual real operator() (const Point& p, real t) const = 0;
	 
  };

}

#endif
