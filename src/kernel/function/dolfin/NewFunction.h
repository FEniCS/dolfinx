// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2005.

#ifndef __NEW_FUNCTION_H
#define __NEW_FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/Point.h>
#include <dolfin/Variable.h>

namespace dolfin
{
  
  class Cell;
  class Mesh;
  class NewFiniteElement;
  class NewVector;
  
  /// This class represents a function defined on a mesh. The function
  /// is defined in terms of a mesh, a finite element and a vector
  /// containing the degrees of freedom of the function on the mesh.

  class NewFunction : public Variable
  {
  public:

    /// Create user-defined function
    NewFunction();

    /// Create function with given degrees of freedom
    NewFunction(const Mesh& mesh, const NewFiniteElement& element, NewVector& x);

    /// Destructor
    virtual ~NewFunction();

    /// Compute projection of function onto a given local finite element space
    void project(const Cell& cell, const NewFiniteElement& element, real c[]) const;

    /// Evaluate function at given point
    virtual real operator() (const Point& p) const;

  private:

    // Collect function data in one place
    class Data
    {
    public:
      Data(const Mesh& mesh, const NewFiniteElement& element, NewVector& x)
	: mesh(mesh), element(element), x(x) {}
      const Mesh& mesh;
      const NewFiniteElement& element;
      NewVector& x;
    };
    
    // Pointer to function data (null if not used)
    Data* data;
    
  };

}

#endif
