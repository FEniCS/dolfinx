// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2007-04-03

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/Vector.h>
#include <dolfin/LocalFunctionData.h>

namespace dolfin
{
  
  class Point;
  class Cell;
  class Vertex;
  class Mesh;
  class AffineMap;
  class FiniteElement;
  class Function;

  /// This class serves as a base class/interface for implementations
  /// of specific function representations.

  class GenericFunction : public ufc::function
  {
  public:

    /// Constructor
    GenericFunction();

    /// Destructor
    virtual ~GenericFunction();

    /// Evaluate function at given point
    virtual real operator() (const Point& point, uint i) = 0;

    /// Evaluate function at given vertex
    virtual real operator() (const Vertex& vertex, uint i) = 0;

    /// Restrict to sub function or component (if possible)
    virtual void sub(uint i) = 0;

    /// Compute interpolation of function onto local finite element space
    virtual void interpolate(real coefficients[], Cell& cell, AffineMap& map, FiniteElement& element) = 0;

    /// Compute interpolation of fsource to local finite element space
    virtual void interpolate(Function& fsource);

    /// Return vector dimension of function
    virtual uint vectordim() const = 0;

    /// Return vector associated with function (if any)
    virtual Vector& vector() = 0;

    /// Return mesh associated with function (if any)
    virtual Mesh& mesh() = 0;

    /// Return element associated with function (if any)
    virtual FiniteElement& element() = 0;

    /// Attach vector to function
    virtual void attach(Vector& x, bool local) = 0;

    /// Attach mesh to function
    virtual void attach(Mesh& mesh, bool local) = 0;

    /// Attach finite element to function
    virtual void attach(FiniteElement& element, bool local) = 0;

    //--- New functions for UFC-based assembly, others may be removed ---
    
    /// Interpolate function on cell
    virtual void interpolate(real* coefficients,
                             const ufc::cell& cell,
                             const ufc::finite_element& finite_element) = 0;

  protected:

    // Local storage for interpolation and evaluation
    LocalFunctionData local;

  };

}

#endif
