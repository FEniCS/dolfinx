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
  class Mapping;
  
  class FunctionSpace {
  public:

	 FunctionSpace(int dim);
	 virtual ~FunctionSpace();

	 // Forward declarations of nested classes
	 class ShapeFunction;
	 class ElementFunction;
	 class Product;

	 // Addition of new shape functions
	 void add(ShapeFunction v);

	 void add(ShapeFunction v, ElementFunction dx);
	 void add(ShapeFunction v, ElementFunction dx, ElementFunction dy);
	 void add(ShapeFunction v, ElementFunction dx, ElementFunction dy, ElementFunction dz);
	 void add(ShapeFunction v, ElementFunction dx, ElementFunction dy, ElementFunction dz, ElementFunction dt);

	 void add(ShapeFunction v, real dx);
	 void add(ShapeFunction v, real dx, real dy);
	 void add(ShapeFunction v, real dx, real dy, real dz);
	 void add(ShapeFunction v, real dx, real dy, real dz, real dt);

	 // Dimension (number of shape functions)
	 int dim() const;
	 
	 // Mapping from local to global degrees of freedom
	 virtual int dof(int i, const Cell &cell) const = 0;
	 
	 // Evaluation of local degree of freedom
	 virtual real dof(int i, const Cell &cell, function f, real t) const = 0;

	 // Update with current mapping
	 void update(const Mapping& mapping);
	 
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

	 // Template for vectors
	 template <class T> class Vector {
	 public:
		
		Vector() {
		  v[0] = 0;
		  v[1] = 0;
		  v[2] = 0;
		}

		Vector(const Vector &v) {
		  this->v[0] = v.v[0];
		  this->v[1] = v.v[1];
		  this->v[2] = v.v[2];
		}

		Vector(const T& v0, const T& v1, const T& v2) {
		  v[0] = &v0;
		  v[1] = &v1;
		  v[2] = &v2;
		}
		
		const T& operator() (int i) {
		  return *v[i];
		}
		
		const T operator, (const Vector& v) const {
		  T w(*this->v[0], *this->v[1], *this->v[2],
				*v.v[0], *v.v[1], *v.v[2]);
		  return w;
		}

	 private:
		const T* v[3];
	 };
	 
	 // Friends
	 friend class Iterator;
	 friend class FiniteElement;
	 
  protected:
	 
	 int _dim;                   // Dimension (number of shape functions)
	 ShortList<ShapeFunction> v; // Shape functions

  };
  
}

#endif
