// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-10-03

#ifndef __COEFFICIENT_H
#define __COEFFICIENT_H

#include <ufc.h>
#include "Data.h"

namespace dolfin
{

  class Cell;
  class FiniteElement;

  /// This class represents a coefficient appearing in a finite
  /// element variational form. A coefficient can be either a
  /// user-defined expression (class Expression), or a function
  /// (class Function) obtained as the solution of a variational
  /// problem.
  ///
  /// This abstract base class defines the interface for all kinds
  /// of coefficients. Sub classes need to implement the restrict()
  /// method which is responsible for computing the expansion
  /// coefficients on a given local element.
  ///
  /// Sub classes may optionally implement the gather() function that
  /// will be called prior to restriction when running in parallel.

  class Coefficient : public ufc::function
  {
  public:

    /// Constructor
    Coefficient();

    /// Destructor
    virtual ~Coefficient();

    /// Evaluate coefficient function
    virtual void eval(double* values, const Data& data) const = 0;

    /// Restrict coefficient to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          int local_facet) const = 0;

    /// Collect off-process coefficients to prepare for interpolation
    virtual void gather() const {}

    /// Convenience function for restriction when facet is unknown
    void restrict(double* w,
                  const FiniteElement& element,
                  const Cell& dolfin_cell,
                  const ufc::cell& ufc_cell) const
    { restrict(w, element, dolfin_cell, ufc_cell, -1); }

    /// Implementation of ufc::function interface
    virtual void evaluate(double* values,
                          const double* coordinates,
                          const ufc::cell& cell) const;

  protected:

    /// Restrict as UFC function (by calling eval)
    void restrict_as_ufc_function(double* w,
                                  const FiniteElement& element,
                                  const Cell& dolfin_cell,
                                  const ufc::cell& ufc_cell,
                                  int local_facet) const;

  private:

    // Function call data
    mutable Data data;

  };

}

#endif
