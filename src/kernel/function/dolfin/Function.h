// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/ElementFunction.h>

namespace dolfin {
  
  class Cell;
  class Mesh;
  class Vector;
  class ElementData;
  class GenericFunction;

  /// A Function represents a function that can be evaluated for given
  /// input. The type of evaluation depends on which type of function
  /// that is represented.

  // FIXME: Needs to be redesigned. Which types do we need, and how
  // FIXME: are they related?
  
  class Function : public Variable {
  public:

    /// Create function specified on given mesh with given nodal values
    Function(Mesh& mesh, dolfin::Vector& x, int dim = 0, int size = 1);

    /// Create function specified by the given expression
    Function(const char* name,  int dim = 0, int size = 1);

    /// Create function specified by the given element data
    Function(ElementData& elmdata);

    /// Destructor
    ~Function();

    // Evaluation for given node and time
    real operator() (const Node&  n, real t = 0.0) const;

    // Evaluation for given point and time
    real operator() (const Point& p, real t = 0.0) const;

    // Evaluation for given coordinates and time
    real operator() (real x, real y, real z, real t) const;

    // Evaluation for given component and time
    real operator() (unsigned int i, real t) const;
    
    // Update function to given time
    void update(real t);

    // Return current time
    real time() const;

    // FIXME: Special member functions below: Should they be removed?
    //---------------------------------------------------------------

    // Get mesh
    Mesh& mesh() const;

    // Update values of element function
    void update(FunctionSpace::ElementFunction& v,
		const FiniteElement& element, const Cell& cell, real t) const;

    // Vector function
    class Vector {
    public:
      
      Vector(Mesh& mesh, dolfin::Vector& x, int size = 3);
      Vector(const char* name,  int size = 3);
      ~Vector();
      
      int size() const;
      
      Function& operator() (int i);
      
    private:
     
      Function** f;
      int _size;

    };

  private:
    
    // Function
    GenericFunction* f;

  };
  
}

#endif
