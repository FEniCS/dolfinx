// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_H
#define __EQUATION_H

#include <dolfin/constants.h>
#include <dolfin/Function.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {

  class FiniteElement;

  typedef FunctionSpace::ShapeFunction ShapeFunction;
  typedef FunctionSpace::Product Product;
  typedef FunctionSpace::ElementFunction ElementFunction;
  
  class Equation {
  public:

	 Equation(int dim);
	 
	 // Variational formulation, left-hand side
	 virtual real lhs(const ShapeFunction &u, const ShapeFunction &v) = 0;
	 // Variational formulation, right-hand side
	 virtual real rhs(const ShapeFunction &v) = 0;

	 // Update before computation of left-hand side
	 void updateLHS(const FiniteElement* element,
						 const Cell*          cell,
						 const Mapping*       mapping,
						 const Quadrature*    quadrature);
	 // Update before computation of right-hand side
	 void updateRHS(const FiniteElement* element,
						 const Cell*          cell,
						 const Mapping*       mapping,
						 const Quadrature*    quadrature);
	 
	 void setTime     (real t);
	 void setTimeStep (real dt);

  protected:

	 // Function data
	 class FunctionPair {
	 public:
		FunctionPair();
		FunctionPair(ElementFunction &v, Function &f);

		// Needed for ShortList
		void operator= (int zero);
		bool operator! () const;

		// Update element function to current element
		void update(const FiniteElement &element,
						const Cell &cell, real t);
		
		ElementFunction *v;
		Function *f;
	 };

	 // Add a function that needs to be updated on every new cell
	 void add(ElementFunction &v, Function &f);

	 // Update equation
	 void update(const FiniteElement* element,
					 const Cell*          cell,
					 const Mapping*       mapping,
					 const Quadrature*    quadrature);
	 
	 // Optional update before computation of left-hand side
	 virtual void updateLHS() {};
	 // Optional update before computation of right-hand side
	 virtual void updateRHS() {};
	 
	 // List of element functions that need to be updated
	 ShortList<FunctionPair> functions;

	 // Integral measures
	 Integral::InteriorMeasure dK;
	 Integral::BoundaryMeasure dS;
	 
	 int dim;  // Space dimension
	 int noeq; // Number of equations
	 real h;   // Mesh size
	 real t;   // Time
	 real dt;  // Time step
	 
  };

}

#endif
