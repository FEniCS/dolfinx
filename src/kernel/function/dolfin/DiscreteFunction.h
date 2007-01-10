// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2006-12-12

#ifndef __DISCRETE_FUNCTION_H
#define __DISCRETE_FUNCTION_H

#include <dolfin/GenericFunction.h>
#include <dolfin/Vector.h>

namespace dolfin
{

  class Mesh;
  
  /// This class implements the functionality for a discrete function
  /// of some given finite element space, defined by a vector of degrees
  /// of freedom, a mesh, and a finite element.

  class DiscreteFunction : public GenericFunction
  {
  public:

    /// Create discrete function from given data
    DiscreteFunction(Vector& x);

    /// Create discrete function from given data
    DiscreteFunction(Vector& x, Mesh& mesh);

    /// Create discrete function from given data
    DiscreteFunction(Vector& x, Mesh& mesh, FiniteElement& element);

    /// Create discrete function from given data (automatically create vector)
    DiscreteFunction(Mesh& mesh, FiniteElement& element);

    /// Copy constructor
    DiscreteFunction(const DiscreteFunction& f);

    /// Destructor
    ~DiscreteFunction();

    /// Evaluate function at given point
    real operator() (const Point& point, uint i);

    /// Evaluate function at given vertex
    real operator() (const Vertex& vertex, uint i);

    // Restrict to sub function or component (if possible)
    void sub(uint i);

    // Copy data from given function
    void copy(const DiscreteFunction& f);

    /// Compute interpolation of function onto local finite element space
    void interpolate(real coefficients[], Cell& cell, AffineMap& map, FiniteElement& element);

    /// Compute interpolation of fsource onto local finite element space.
    void interpolate(Function& fsource);

    /// Return vector dimension of function
    uint vectordim() const;

    /// Return vector associated with function (if any)
    Vector& vector();

    /// Return mesh associated with function (if any)
    Mesh& mesh();

    /// Return element associated with function (if any)
    FiniteElement& element();

    /// Attach vector to function
    void attach(Vector& x, bool local);

    /// Attach mesh to function
    void attach(Mesh& mesh, bool local);

    /// Attach finite element to function
    void attach(FiniteElement& element, bool local);

    /// Reinitialize to given data (automatically create vector)
    void init(Mesh& mesh, FiniteElement& element);

  private:

    // Update vector dimension from current element
    void updateVectorDimension();

    // Pointer to degrees of freedom
    Vector* _x;

    // Pointer to mesh
    Mesh* _mesh;

    // Pointer to finite element
    FiniteElement* _element;
    
    // Number of vector dimensions
    uint _vectordim;

    // Current component
    uint component;

    // Current offset for mixed sub function
    uint mixed_offset;
    
    // Current component offset
    uint component_offset;

    // True if vector is local (not a reference to another vector)
    bool vector_local;

    // True if mesh is local (not a reference to another mesh)
    bool mesh_local;

    // True if element is local (not a reference to another element)
    bool element_local;

  };

}

#endif
