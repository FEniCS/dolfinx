// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-28
// Last changed: 2009-10-04

#ifndef __COEFFICIENT_ASSIGNER_H
#define __COEFFICIENT_ASSIGNER_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Form;
  class GenericFunction;

  /// This class is used for assignment of coefficients to
  /// forms, which allows magic like
  ///
  ///   a.f = f
  ///   a.g = g
  ///
  /// which will insert the coefficients f and g in the correct
  /// positions in the list of coefficients for the form.

  class CoefficientAssigner
  {
  public:

    /// Create coefficient assigner for coefficient with given number
    CoefficientAssigner(Form& form, uint number);

    /// Destructor
    ~CoefficientAssigner();

    /// Assign coefficient
    void operator= (const GenericFunction& coefficient);

  private:

    // The form
    Form& form;

    // The number of the coefficient
    uint number;

  };

}

#endif
