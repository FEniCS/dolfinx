// Copyright (C) 2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-05-08
// Last changed: 2008-10-30

#ifndef __UFC_FUNCTION_H
#define __UFC_FUNCTION_H

#include <ufc.h>

namespace dolfin
{

  /// This is a utility class used by Function to wrap callbacks from
  /// ufc::finite_element::evaluate_dof to ufc::function::evaluate.

  class UFCFunction : public ufc::function
  {
  public:

    /// Create wrapper for given function
    UFCFunction(const Function& v);

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(double* values,
                  const double* coordinates,
                  const ufc::cell& cell) const;

  private:

    // The function
    const Function& v;

  };

}

#endif
