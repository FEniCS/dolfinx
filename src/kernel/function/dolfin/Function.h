// Copyright (C) 2003-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-11-28
// Last changed: 2005

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/Variable.h>

namespace dolfin
{

  class Point;
  class Node;
  class Cell;
  class Mesh;
  class Vector;
  class AffineMap;
  class FiniteElement;
  
  /// This class represents a function defined on a mesh. The function
  /// is defined in terms of a mesh, a finite element and a vector
  /// containing the degrees of freedom of the function on the mesh.

  /// Vector functions are initialized by a vector x containing the 
  /// function values for all three vector components and no_comp, 
  /// the number of vector components.  
  /// The function values in x should be ordered as: 
  /// vector component j at dof i is listed at i*no_comp+j.   

  class Function : public Variable
  {
  public:

    /// Create user-defined function
    Function();

    /// Create a function (choose mesh and element automatically)
    Function(Vector& x);

    /// Create a function (choose element automatically)
    Function(Vector& x, Mesh& mesh);

    /// Create a function
    Function(Vector& x, Mesh& mesh, const FiniteElement& element);

    /// Destructor
    virtual ~Function();

    /// Compute interpolation of function onto the local finite element space
    void interpolate(real coefficients[], const AffineMap& map);

    /// Evaluate function at given node
    real operator() (const Node& node) const;

    /// Evaluate function at given node
    real operator() (const Node& node, uint i) const;

    /// Evaluate function at given point
    virtual real operator() (const Point& point) const;

    /// Evaluate vector-valued function at given point
    virtual real operator() (const Point& point, uint i) const;

    /// Return the mesh on which the function is defined
    Mesh& mesh();

    /// Return the finite element defining the function space
    const FiniteElement& element() const;

    /// Return current time
    real time() const;

    /// Specify current time
    void set(real time);

    /// Specify finite element
    void set(const FiniteElement& element);

  private:

    // Pointer to degrees of freedom
    Vector* _x;

    // Pointer to finite element
    const FiniteElement* _element;

    // Current time
    real t;

    // Temporary data used for interpolation
    int* dofs;
    uint* components;
    Point* points;
    
  protected:

    // Pointer to mesh
    Mesh* _mesh;

    // Pointer to current cell (for user-defined functions)
    const Cell* _cell;

  };

}

#endif
