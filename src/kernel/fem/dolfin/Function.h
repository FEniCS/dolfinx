// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/function.h>
#include <dolfin/Vector.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {

  class Cell;
  class Grid;
  class GenericFunction;
  
  class Function : public Variable {
  public:

	 Function(Grid &grid, Vector &x);
	 Function(Grid &grid, const char *name);

	 // Update values of element function
	 void update(FunctionSpace::ElementFunction &v,
					 const FiniteElement &element, const Cell &cell, real t) const;

	 // Evaluation of function
	 real operator() (const Node&  n, real t = 0.0) const;
	 real operator() (const Point& p, real t = 0.0) const;
	 
	 // Get grid
	 const Grid& grid() const;

  private:

	 // Grid
	 Grid& _grid;

	 // Function
	 GenericFunction* f;

  };

}

#endif
