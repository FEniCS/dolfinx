// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <dolfin/ShortList.h>
#include <dolfin/function.h>

namespace dolfin {

  class FunctionSpace {
  public:

	 FunctionSpace(int dim);
	 ~FunctionSpace();

	 // Forward declarations of nested classes
	 class ShapeFunction;
	 class ElementFunction;
	 class Product;
	 
	 // Addition of new shape functions
	 int add(ShapeFunction v);
	 int add(ShapeFunction v, ElementFunction dx);
	 int add(ShapeFunction v, ElementFunction dx, ElementFunction dy);
	 int add(ShapeFunction v, ElementFunction dx, ElementFunction dy, ElementFunction dz);
	 int add(ShapeFunction v, ElementFunction dx, ElementFunction dy, ElementFunction dz,	ElementFunction dt);
	 
  protected:
	 
	 int dim;          // Dimension (number of shape functions)
	 ShapeFunction *v; // Shape functions

  private:

	 int current;      // Current function (used when adding functions)
	 
  };

}

#endif
