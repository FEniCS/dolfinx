// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHAPE_FUNCTION_H
#define __SHAPE_FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/function.h>
#include <dolfin/FunctionSpace.h>

namespace dolfin {

  class FunctionSpace::Product;
  class FunctionSpace::ElementFunction;
  
  class FunctionSpace::ShapeFunction {
  public:

	 // Empty constructor for v = 0
	 ShapeFunction();
	 
	 // Constructor for v = 1
	 ShapeFunction(int i);

	 // Initialisation
	 ShapeFunction(function f);
	 ShapeFunction(const ShapeFunction &v);

	 // Specification of derivatives
	 void set(ElementFunction dx, ElementFunction dy, ElementFunction dz, ElementFunction dt);
	 
	 // Derivatives
	 ElementFunction dx() const;
	 ElementFunction dy() const;
	 ElementFunction dz() const;
	 ElementFunction dt() const;
	 
	 //--- Operators ---

	 // Evaluation
	 real operator() (real x, real y, real z, real t) const;

	 // Assignment
	 ShapeFunction& operator= (const ShapeFunction &v);

	 // Addition
	 ElementFunction operator+ (const ShapeFunction   &v) const;
	 ElementFunction operator+ (const Product         &v) const;
	 ElementFunction operator+ (const ElementFunction &v) const;

	 // Subtraction
	 ElementFunction operator- (const ShapeFunction   &v) const;
	 ElementFunction operator- (const Product         &v) const;
	 ElementFunction operator- (const ElementFunction &v) const;

	 // Multiplication
	 Product         operator* (const ShapeFunction   &v) const;
	 Product         operator* (const Product         &v) const;
	 ElementFunction operator* (const ElementFunction &v) const;
	 ElementFunction operator* (const real a) const;

	 // Friends
	 friend FunctionSpace;
	 
  private:

	 int id;

  };

}

#endif
