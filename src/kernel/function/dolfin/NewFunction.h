// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FUNCTION_H
#define __NEW_FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/Variable.h>

namespace dolfin
{
  
  class Cell;
  class Mesh;
  class Point;
  class NewFiniteElement;
  class NewVector;
  
  /// This class represents a function defined on a mesh. The function
  /// is defined in terms of a mesh, a finite element and a vector
  /// containing the degrees of freedom of the function on the mesh.

  class NewFunction : public Variable
  {
  public:

    /// Create function on a given mesh
    NewFunction(const Mesh& mesh, const NewFiniteElement& element, NewVector& x);

    /// Destructor
    virtual ~NewFunction();

    /// Compute projection onto a given local finite element space
    void project(const Cell& cell, const NewFiniteElement& element, real c[]) const;

    /// Point evaluation
    virtual real operator()(const Point& p);

  private:

    // The mesh
    const Mesh& mesh;

    // The finite element
    const NewFiniteElement& element;

    // The vector containg the degrees of freedom
    NewVector& x;
    
  };

}

#endif
