// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-09-28
// Last changed: 2009-12-11

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <ufc.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class FiniteElement;

  /// This is a common base class for functions. Functions can be
  /// evaluated at a given point and they can be restricted to a given
  /// cell in a finite element mesh. This functionality is implemented
  /// by sub-classes that implement the eval() and restrict() functions.
  ///
  /// DOLFIN provides two implementations of the GenericFunction
  /// interface in the form of the classes Function and Expression.
  ///
  /// Sub-classes may optionally implement the gather() function that
  /// will be called prior to restriction when running in parallel.

  class GenericFunction : public ufc::function, public Variable
  {
  public:

    /// Constructor
    GenericFunction();

    /// Destructor
    virtual ~GenericFunction();

    //--- Functions that must be implemented by sub-classes ---

    /// Return value rank
    virtual uint value_rank() const = 0;

    /// Return value dimension for given axis
    virtual uint value_dimension(uint i) const = 0;

    /// Evaluate function for given data
    virtual void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const = 0;

    /// Restrict function to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          int local_facet) const = 0;

    /// Compute values at all mesh vertices
    virtual void compute_vertex_values(Array<double>& vertex_values,
                                       const Mesh& mesh) const = 0;

    //--- Optional functions to be implemented by sub-classes ---

    /// Collect off-process coefficients to prepare for interpolation
    virtual void gather() const {}

    //--- Convenience functions ---

    /// Return value size (product of value dimensions)
    uint value_size() const;

    /// Convenience function for restriction when facet is unknown
    void restrict(double* w,
                  const FiniteElement& element,
                  const Cell& dolfin_cell,
                  const ufc::cell& ufc_cell) const
    { restrict(w, element, dolfin_cell, ufc_cell, -1); }

    //--- Implementation of ufc::function interface ---

    /// Evaluate function at given point in cell
    virtual void evaluate(double* values,
                          const double* coordinates,
                          const ufc::cell& cell) const;

  protected:

    // Restrict as UFC function (by calling eval)
    void restrict_as_ufc_function(double* w,
                                  const FiniteElement& element,
                                  const Cell& dolfin_cell,
                                  const ufc::cell& ufc_cell,
                                  int local_facet) const;

  };

}

#endif
