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
      
      const FunctionSpace &V;
      Array<ShapeFunction>::Iterator v;
      
    };
    
    // Friends
    friend class Iterator;
    friend class FiniteElement;
    
  protected:
    
    int _dim;               // Dimension (number of shape functions)
    Array<ShapeFunction> v; // Shape functions
    
  };
  
}

#endif
