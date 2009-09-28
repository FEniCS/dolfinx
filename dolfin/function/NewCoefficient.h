// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-09-28

#ifndef __NEW_COEFFICIENT_H
#define __NEW_COEFFICIENT_H

#include <ufc.h>

namespace dolfin
{

  class Cell;
  class FunctionSpace;

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

  // FIXME: Rename this class to Coefficient after renaming/reworking
  // FIXME: the current Coefficient class.

  class NewCoefficient
  {
  public:

    /// Constructor
    NewCoefficient() {}

    /// Destructor
    virtual ~NewCoefficient() {}

    /// Restrict coefficient to local element (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          const FunctionSpace& V,
                          int local_facet) const = 0;

    /// Restrict coefficient to local facet (compute expansion coefficients w)
    inline void restrict(double* w,
                         const Cell& dolfin_cell,
                         const ufc::cell& ufc_cell,
                         const FunctionSpace& V)
    { restrict(w, dolfin_cell, ufc_cell, V, -1); }

  };

}

#endif
