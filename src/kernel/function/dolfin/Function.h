// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FunctionPointer.h>
#include <dolfin/NewArray.h>

namespace dolfin {
  
  class Cell;
  class Mesh;
  class Vector;
  class ElementData;
  class NewPDE;
  class GenericFunction;

  /// A Function represents a function that can be evaluated for given
  /// input. The type of evaluation depends on which type of function
  /// that is represented.

  // FIXME: Needs to be redesigned. Which types do we need, and how
  // FIXME: are they related?
  
  class Function : public Variable {
  public:

    /// Create a function of unspecified type
    Function();

    /// Create function specified on given mesh with given nodal values
    Function(Mesh& mesh, dolfin::Vector& x, int dim = 0, int size = 1);

    /// Create function specified by the given expression
    Function(const char* name,  int dim = 0, int size = 1);

    /// Create function from function pointer
    Function(function fp);

    /// Create an ODE function of given dimension
    Function(unsigned int N);

    /// Destructor
    ~Function();

    /// Initialize function on given mesh with given nodal values
    void init(Mesh& mesh, dolfin::Vector& x, int dim = 0, int size = 1);
    
    /// Create function specified by the given expression
    void init(const char* name,  int dim = 0, int size = 1);
    
    /// Create function from function pointer
    void init(function fp);

    /// Create an ODE function of given dimension
    void init(unsigned int N);
    
    /// Evaluation for given node and time
    real operator() (const Node&  n, real t = 0.0) const;
    real operator() (const Node&  n, real t = 0.0);
    
    /// Evaluation for given point and time
    real operator() (const Point& p, real t = 0.0) const;
    real operator() (const Point& p, real t = 0.0);
    
    /// Evaluation for given coordinates and time
    real operator() (real x, real y, real z, real t) const;
    real operator() (real x, real y, real z, real t);
    
    /// Evaluation for given component and time
    real operator() (unsigned int i, real t) const;
    real operator() (unsigned int i, real t);
    
    /// Absolute values 
    real abs(const Node&  n, real t = 0.0);
    real abs(const Point& p, real t = 0.0);
    real abs(real x, real y, real z, real t);
    real abs(unsigned int i, real t);
    
    // Update function to given time
    void update(real t);
    
    // Return current time
    real time() const;
    
    // FIXME: Special member functions below: Should they be removed?
    //---------------------------------------------------------------
    
    // Get mesh
    Mesh& mesh() const;
    
    // Get element data
    ElementData& elmdata();
    
    // Update values of element function
    void update(FunctionSpace::ElementFunction& v,
		const FiniteElement& element, const Cell& cell, real t) const;
    
    // Update local function (restriction to given cell)
    void update(NewArray<real>& w, const Cell& cell, const NewPDE& pde) const;

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
