// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PRODUCT_H
#define __PRODUCT_H

#include <dolfin/constants.h>
#include <dolfin/FunctionSpace.h>

namespace dolfin {

  class FunctionSpace::ShapeFunction;
  class FunctionSpace::ElementFunction;
  
  class FunctionSpace::Product {
  public:
	 
	 // Empty constructor for v = 1
	 Product();

	 // Initialisation
	 Product(const ShapeFunction &v);
	 Product(const Product       &v);
	 
	 // Constructors for v0 * v1
	 Product(const ShapeFunction &v0, const ShapeFunction &v1);
	 Product(const Product       &v0, const Product       &v1);
	 Product(const ShapeFunction &v0, const Product       &v1);
	 
	 // Destructor
	 ~Product();
	 
	 // Assignment (v0 * v1)
	 void set(const ShapeFunction &v0, const ShapeFunction &v1);
	 void set(const Product       &v0, const Product       &v1);
	 void set(const ShapeFunction &v0, const Product       &v1);
	 
	 //--- Operators ---
	 
	 // Evaluation
	 real operator() (real x, real y, real z, real t);
	 
	 // Assignment
	 Product& operator= (const ShapeFunction &v);
	 Product& operator= (const Product       &v);
	 
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
	 
  private:
	 
	 int n;            // Number of factors
	 ShapeFunction *v; // Shape functions
	 
  };
  
}

#endif
