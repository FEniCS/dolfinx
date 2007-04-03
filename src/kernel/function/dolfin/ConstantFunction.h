// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2007-04-02

#ifndef __CONSTANT_FUNCTION_H
#define __CONSTANT_FUNCTION_H

#include <dolfin/GenericFunction.h>

namespace dolfin
{
  /// This class implements the functionality for a function
  /// with a constant value.

  class ConstantFunction : public GenericFunction
  {
  public:

    /// Create function from given value
    ConstantFunction(real value);

    /// Copy constructor
    ConstantFunction(const ConstantFunction& f);

    /// Destructor
    ~ConstantFunction();

    /// Evaluate function at given point
    real operator() (const Point& point, uint i);

    /// Evaluate function at given vertex
    real operator() (const Vertex& vertex, uint i);

    // Restrict to sub function or component (if possible)
    void sub(uint i);

    /// Interpolate function to finite element basis
    void interpolate(real coefficients[],
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element) const;

    /// Compute interpolation of function onto local finite element space
    void interpolate(real coefficients[], Cell& cell, AffineMap& map, FiniteElement& element);

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

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(real* values,
                  const real* coordinates,
                  const ufc::cell& cell) const;

  private:
    
    // Value of constant function
    real value;

    // Pointer to mesh associated with function (null if none)
    Mesh* _mesh;
    
    // True if mesh is local (not a reference to another mesh)
    bool mesh_local;

  };

}

#endif
