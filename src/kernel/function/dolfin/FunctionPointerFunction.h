// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-02

#ifndef __FUNCTION_POINTER_FUNCTION_H
#define __FUNCTION_POINTER_FUNCTION_H

#include <ufc.h>

#include <dolfin/FunctionPointer.h>
#include <dolfin/GenericFunction.h>

namespace dolfin
{
  /// This class implements the functionality for a user-defined
  /// function given by a function pointer.

  class FunctionPointerFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Create function from function pointer
    FunctionPointerFunction(FunctionPointer f, uint vectordim);

    /// Copy constructor
    FunctionPointerFunction(const FunctionPointerFunction& f);

    /// Destructor
    ~FunctionPointerFunction();

    /// Evaluate function at given point
    real operator() (const Point& point, uint i);

    /// Evaluate function at given vertex
    real operator() (const Vertex& vertex, uint i);

    /// Restrict to sub function or component (if possible)
    void sub(uint i);

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
    
    // Function pointer to user-defined function
    FunctionPointer f;

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
