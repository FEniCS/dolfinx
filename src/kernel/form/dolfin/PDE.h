// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Fredrik Bengzon and Johan Jansson, 2004.

#ifndef __PDE_H
#define __PDE_H

#include <dolfin/constants.h>
#include <dolfin/List.h>
#include <dolfin/Map.h>
#include <dolfin/Function.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/FiniteElement.h>

namespace dolfin {
  
  typedef FunctionSpace::ShapeFunction ShapeFunction;
  typedef FunctionSpace::Product Product;
  typedef FunctionSpace::ElementFunction ElementFunction;
  
  class PDE {
  public:
 
    /// Constructor for PDE with given space dimension and system size
    PDE(int dim, int noeq = 1);

    /// Destructor
    virtual ~PDE();
    
    /// Variational formulation, left-hand side
    virtual real lhs(const ShapeFunction &u, const ShapeFunction &v)
    {
      return 0.0;
    }

    /// Variational formulation, right-hand side
    virtual real rhs(const ShapeFunction &v)
    {
      return 0.0;
    }
    
    /// Variational formulation for systems, left-hand side
    virtual real lhs(ShapeFunction::Vector &u,
 		     ShapeFunction::Vector &v)
    {
      return lhs(u(0), v(0));
    }
        
    /// Variational formulation for systems, right-hand side
    virtual real rhs(ShapeFunction::Vector &v)
    {
      return rhs(v(0));
    }
    
    /// Update before computation of left-hand side
    void updateLHS(FiniteElement::Vector& element,
                   const Cell& cell,
                   const Map& mapping,
                   const Quadrature& quadrature);
    
    /// Update before computation of right-hand side
    void updateRHS(FiniteElement::Vector& element,
                   const Cell& cell,
                   const Map& mapping,
                   const Quadrature& quadrature);

    /// Return number of equations
    int size();

    ///Public data
    real t; // Time
    real k; // Time step
    
  protected:
    
    // --- Derivatives (computed using map)
    
    // Derivative of constant
    real ddx(real a) const;
    real ddy(real a) const;
    real ddz(real a) const;
    real ddt(real a) const;
    
    // Derivative of ShapeFunction
    const ElementFunction& ddx(const ShapeFunction& v) const;
    const ElementFunction& ddy(const ShapeFunction& v) const;
    const ElementFunction& ddz(const ShapeFunction& v) const;
    const ElementFunction& ddt(const ShapeFunction& v) const;
    
    // Derivative of Product
    const ElementFunction ddx(const Product& v) const;
    const ElementFunction ddy(const Product& v) const;
    const ElementFunction ddz(const Product& v) const;
    const ElementFunction ddt(const Product& v) const;
    
    // Derivative of ElementFunction
    const ElementFunction ddx(const ElementFunction& v) const;
    const ElementFunction ddy(const ElementFunction& v) const;
    const ElementFunction ddz(const ElementFunction& v) const;
    const ElementFunction ddt(const ElementFunction& v) const;
    
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
    void update(FiniteElement::Vector& element,
		const Cell& cell,
		const Map& map,
		const Quadrature& quadrature);
    
    // Optional update before computation of left-hand side
    virtual void updateLHS() {};
    // Optional update before computation of right-hand side
    virtual void updateRHS() {};
    
    // List of element functions that need to be updated
    List<FunctionPair> functions;
    
    // Map from reference element
    const Map* map;
    
    // Cell 
    const Cell* cell;
    
    // Integral measures
    Integral::InteriorMeasure dx;
    Integral::BoundaryMeasure ds;
    
    int dim;  // Space dimension
    int noeq; // Number of equations
    
    real h; // Local mesh size

  };
  
}

#endif
