// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Definition of a finite element. This is a modified version
// or the standard definition given by Ciarlet (1976).
//                       
//   1.   A reference cell K0
//
//   2.a) A function space P on the reference cell (the local trial space)
//     b) A mapping I from local to global degrees of freedom
//
//   3.a) A function space Q on the reference cell (the local test space)
//     b) A mapping J from local to global degrees of freedom

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <dolfin/constants.h>
#include <dolfin/ShortList.h>
#include <dolfin/FunctionSpace.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {

  class ExpressionFunction;

  class FiniteElement {
  public:
    
    // Constructor
    FiniteElement(FunctionSpace& trial, FunctionSpace& test) : P(trial), Q(test) {};
    
    // Dimension (of trial space)
    int dim() const;
    
    // Update function spaces
    void update(const Mapping* mapping);
    
    // Iterator over shape functions in the local trial space
    class TrialFunctionIterator {
    public:
      
      TrialFunctionIterator(const FiniteElement& element);
      TrialFunctionIterator(const FiniteElement* element);
      
      // Global dof (mapping I)
      int dof(const Cell& cell) const;
      
      // Evaluation of dof
      real dof(const Cell& cell, const ExpressionFunction& f, real t) const;
      
      int  index() const;
      bool end() const;
      void operator++();
      operator FunctionSpace::ShapeFunction() const;
      FunctionSpace::ShapeFunction& operator*() const;
      FunctionSpace::ShapeFunction* operator->() const;
      
    private:
      
      const FiniteElement& e;
      FunctionSpace::Iterator v;
      
    };
    
    // Iterator over shape functions in the local test space
    class TestFunctionIterator {
    public:
      
      TestFunctionIterator(const FiniteElement& element);
      TestFunctionIterator(const FiniteElement* element);
      
      // Global dof (mapping J)
      int dof(const Cell& cell) const;
      
      // Evaluation of dof
      real dof(const Cell& cell, const ExpressionFunction& f, real t) const;
      
      int  index() const;                               // Index in list
      bool end() const;                                 // End of list?
      void operator++();                                // Increment
      operator FunctionSpace::ShapeFunction() const;    // Cast to ShapeFunction
      FunctionSpace::ShapeFunction& operator*() const;  // Dereferencing
      FunctionSpace::ShapeFunction* operator->() const; // -> access
      
    private:
      
      const FiniteElement& e;
      FunctionSpace::Iterator v;
      
    };
    
    // Friends
    friend class TrialFunctionIterator;
    friend class TestFunctionIterator;
    
  private:
    
    FunctionSpace& P; // Local trial space on reference cell
    FunctionSpace& Q; // Local test space on reference cell
    
  };
  
}

#endif
