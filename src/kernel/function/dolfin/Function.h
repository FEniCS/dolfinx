// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {
  
  class Cell;
  class Mesh;
  class GenericFunction;
  class Vector;
  
  class Function : public Variable {
  public:
    
    Function(Mesh& mesh, dolfin::Vector& x, int dim = 0, int size = 1);
    Function(Mesh& mesh, const char* name,  int dim = 0, int size = 1);
    ~Function();
    
    // Update values of element function
    void update(FunctionSpace::ElementFunction& v,
		const FiniteElement& element, const Cell& cell, real t) const;
    
    // Evaluation of function
    real operator() (const Node&  n, real t = 0.0) const;
    real operator() (const Point& p, real t = 0.0) const;
    
    // Get mesh
    Mesh& mesh() const;

    // Time value
    real t;

    // Vector function
    class Vector {
    public:
      
      Vector(Mesh& mesh, dolfin::Vector& x, int size = 3);
      Vector(Mesh& mesh, const char* name,  int size = 3);
      ~Vector();
      
      int size() const;
      
      Function& operator() (int i);
      
    private:
      Function** f;
      int _size;
    };

  private:
    
    // Mesh
    Mesh& _mesh;
    
    // Function
    GenericFunction* f;

  };
  
}

#endif
