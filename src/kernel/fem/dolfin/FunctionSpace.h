// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <dolfin/Cell.h>
#include <dolfin/ShortList.h>
#include <dolfin/function.h>
#include <dolfin/shapefunctions.h>

namespace dolfin {

  class FiniteElement;
  
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
	 int add(ShapeFunction v, ElementFunction dx, ElementFunction dy, ElementFunction dz, ElementFunction dt);

	 int add(ShapeFunction v, real dx);
	 int add(ShapeFunction v, real dx, real dy);
	 int add(ShapeFunction v, real dx, real dy, real dz);
	 int add(ShapeFunction v, real dx, real dy, real dz, real dt);

	 // Dimension (number of shape functions)
	 int dim() const;
	 
	 // Iterator for shape functions in the function space
	 class Iterator {
	 public:
		
		Iterator(const FunctionSpace &functionSpace);
		
		int  dof(const Cell &cell) const;
		real dof(const Cell &cell, function f, real t) const;

		int  index() const;
		bool end() const;
		void operator++();
		ShapeFunction* pointer() const;
		ShapeFunction& operator*() const;
		ShapeFunction* operator->() const;
		
	 private:

		const FunctionSpace &V;
		ShortList<ShapeFunction>::Iterator v;
		
	 };

	 // Mapping from local to global degrees of freedom
	 virtual int dof(int i, const Cell &cell) const = 0;
	 
	 // Evaluation of local degree of freedom
	 virtual real dof(int i, const Cell &cell, function f, real t) const = 0;
	 
	 // Friends
	 friend class Iterator;
	 friend class FiniteElement;
	 
  protected:
	 
	 int _dim;                   // Dimension (number of shape functions)
	 ShortList<ShapeFunction> v; // Shape functions

  };
  
}

#endif
