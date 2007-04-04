// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-04

#ifndef __NON_MATCHING_FUNCTION_H
#define __NON_MATCHING_FUNCTION_H

#include <dolfin/FiniteElement.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/GenericFunction.h>

namespace dolfin
{
  class Function;

  /// This class implements the functionality for a non-matching
  /// discrete function, that is a discrete function defined on a
  /// finite element space A, but evaluated (typically projected) on a
  /// finite element space B.

  class NonMatchingFunction : public GenericFunction
  {
  public:

    /// Create non-matching function
    NonMatchingFunction(DiscreteFunction& F);

    /// Copy constructor
    NonMatchingFunction(const NonMatchingFunction& f);

    /// Destructor
    ~NonMatchingFunction();

    /// Evaluate function at given point
    real operator() (const Point& point, uint i);

    /// Evaluate function at given vertex
    real operator() (const Vertex& vertex, uint i);

    /// Restrict to sub function or component (if possible)
    void sub(uint i);
    
    /// Compute interpolation of function onto local finite element space
    void interpolate(real coefficients[], Cell& cell, AffineMap& map,
		     FiniteElement& element);

    /// Return vector dimension of function
    uint vectordim() const;

    /// Calling this function generates an error (no vector associated)
    Vector& vector();

    /// Return mesh associated with function (if any)
    Mesh& mesh();

    /// Calling this function generates an error (no element associated)
    FiniteElement& element();

    /// Calling this function generates an error (no vector can be attached)
    void attach(Vector& x, bool local);

    /// Attach mesh to function
    void attach(Mesh& mesh, bool local);

    /// Calling this function generates an error (no element can be attached)
    void attach(FiniteElement& element, bool local);

    //--- New functions for UFC-based assembly, others may be removed ---

    /// Interpolate function on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

  private:

    // Pointer to Function with overloaded evaluation operator
    DiscreteFunction* F;

    // Number of vector dimensions
    uint _vectordim;

    // Current component
    uint component;

    // Pointer to mesh associated with function (null if none)
    Mesh* _mesh;

    // True if mesh is local (not a reference to another mesh)
    bool mesh_local;
    
  };

}

#endif
