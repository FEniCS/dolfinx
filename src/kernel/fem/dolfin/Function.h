// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/function.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {

  class Vector;
  class Cell;
  class Grid;
  
  class Function {
  public:

	 enum Representation { DOF, FUNCTION };
	 
	 Function(Grid &grid, Vector &x);
	 Function(Grid &grid, const char *function);

	 // Update values of element function
	 void update(FunctionSpace::ElementFunction &v,
					 const FiniteElement &element, const Cell &cell, real t) const;
	 
  private:

	 Grid *grid;

	 // Type of representation
	 Representation representation;

	 // Function data
	 Vector *x;  // Values given by a vector of nodal values
	 function f; // Values given by a function pointer
	 
  };

}

#endif
