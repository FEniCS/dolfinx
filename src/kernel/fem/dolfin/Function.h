// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/function.h>
#include <dolfin/Vector.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {
  
  class Cell;
  class Grid;
  class GenericFunction;
  
  class Function : public Variable {
  public:
    
    Function(Grid& grid, dolfin::Vector& x, int dim = 0, int size = 1);
    Function(Grid& grid, const char* name,  int dim = 0, int size = 1);
    ~Function();
    
    // Update values of element function
    void update(FunctionSpace::ElementFunction& v,
		const FiniteElement& element, const Cell& cell, real t) const;
    
    // Evaluation of function
    real operator() (const Node&  n, real t = 0.0) const;
    real operator() (const Point& p, real t = 0.0) const;
    
    // Get grid
    Grid& grid() const;

    // Time value
    real t;

    // Vector function
    class Vector {
    public:
      
      Vector(Grid& grid, dolfin::Vector& x, int size = 3);
      Vector(Grid& grid, const char* name,  int size = 3);
      ~Vector();
      
      int size() const;
      
      Function& operator() (int i);
      
    private:
      Function** f;
      int _size;
    };

  private:
    
    // Grid
    Grid& _grid;
    
    // Function
    GenericFunction* f;

  };
  
}

#endif
