// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <dolfin/Cell.h>
#include <dolfin/Array.h>
#include <dolfin/shapefunctions.h>

namespace dolfin {

  class ExpressionFunction;
  class FiniteElement;
  class Map;
  
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
    
    void add(ShapeFunction v, ElementFunction ddx);
    void add(ShapeFunction v, ElementFunction ddx, ElementFunction ddy);
    void add(ShapeFunction v, ElementFunction ddx, ElementFunction ddy, ElementFunction ddz);
    void add(ShapeFunction v, ElementFunction ddx, ElementFunction ddy, ElementFunction ddz, ElementFunction ddt);
    
    void add(ShapeFunction v, real ddx);
    void add(ShapeFunction v, real ddx, real ddy);
    void add(ShapeFunction v, real ddx, real ddy, real ddz);
    void add(ShapeFunction v, real ddx, real ddy, real ddz, real ddt);
    
    // Dimension (number of shape functions)
    int dim() const;
    
    // Map from local to global degrees of freedom
    virtual int dof(int i, const Cell& cell) const = 0;
    
    // Evaluation of local degree of freedom
    virtual real dof(int i, const Cell& cell, const ExpressionFunction& f, real t) const = 0;
    
    // Update with current map
    void update(const Map& map);
    
    // Iterator for shape functions in the function space
    class Iterator {
    public:
		
      Iterator(const FunctionSpace &functionSpace);
      
      int  dof(const Cell& cell) const;
      real dof(const Cell& cell, const ExpressionFunction& f, real t) const;
      
      int  index() const;
      bool end() const;
      void operator++();
      ShapeFunction* pointer() const;
      ShapeFunction& operator*() const;
      ShapeFunction* operator->() const;
      
    private:
      
      const FunctionSpace *V;
      Array<ShapeFunction>::Iterator v;
      
    };
    
    // Friends
    friend class Iterator;
    friend class FiniteElement;

    // Vector function space
    class Vector {
    public:
      
      Vector(int size = 3);
      Vector(const Vector& v);
      //Vector(const FunctionSpace& v0, const FunctionSpace& v1, const FunctionSpace& v2);
      ~Vector();
      
      int size() const;
      
      FunctionSpace& operator() (int i);
      
    private:
      FunctionSpace** v;
      int _size;
    };
    
  protected:
    
    int _dim;               // Dimension (number of shape functions)
    Array<ShapeFunction> v; // Shape functions

  };
  
}

#endif

