// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PDE_H
#define __PDE_H

#include <dolfin/constants.h>
#include <dolfin/List.h>
#include <dolfin/Mapping.h>
#include <dolfin/Function.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {
  
  class FiniteElement;
  
  typedef FunctionSpace::ShapeFunction ShapeFunction;
  typedef FunctionSpace::Product Product;
  typedef FunctionSpace::ElementFunction ElementFunction;
  
  class PDE {
  public:
    
    PDE(int dim);
    virtual ~PDE();
    
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
    
    // Public data
    real h; // Mesh size
    real t; // Time
    real k; // Time step
    
  protected:
    
    // --- Derivatives (computed using mapping)
    
    // Derivative of constant
    real dx(real a) const;
    real dy(real a) const;
    real dz(real a) const;
    real dt(real a) const;
    
    // Derivative of ShapeFunction
    const ElementFunction& dx(const ShapeFunction& v) const;
    const ElementFunction& dy(const ShapeFunction& v) const;
    const ElementFunction& dz(const ShapeFunction& v) const;
    const ElementFunction& dt(const ShapeFunction& v) const;
    
    // Derivative of Product
    const ElementFunction dx(const Product& v) const;
    const ElementFunction dy(const Product& v) const;
    const ElementFunction dz(const Product& v) const;
    const ElementFunction dt(const Product& v) const;
    
    // Derivative of ElementFunction
    const ElementFunction dx(const ElementFunction& v) const;
    const ElementFunction dy(const ElementFunction& v) const;
    const ElementFunction dz(const ElementFunction& v) const;
    const ElementFunction dt(const ElementFunction& v) const;
    
    // Gradients
    const FunctionSpace::ElementFunction::Vector grad(const ShapeFunction& v);
    
    // Function data
    class FunctionPair {
    public:
      FunctionPair();
      FunctionPair(ElementFunction& v, Function& f);
      
      // Update element function to current element
      void update(const FiniteElement& element,
		  const Cell& cell, real t);
      
      ElementFunction* v;
      Function* f;
    };
    
    // Add a function that needs to be updated on every new cell
    void add(ElementFunction& v, Function& f);
    void add(ElementFunction::Vector &v, Function::Vector &f);
    
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
    List<FunctionPair> functions;
    
    // Mapping from reference element
    const Mapping* mapping;
    
    // Integral measures
    Integral::InteriorMeasure dK;
    Integral::BoundaryMeasure dS;
    
    int dim;  // Space dimension
    int noeq; // Number of equations
    
  };
  
}

#endif
