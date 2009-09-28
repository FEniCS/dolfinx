// Copyright (C) 2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-05-08
// Last changed: 2009-09-28

#ifndef __UFC_FUNCTION_H
#define __UFC_FUNCTION_H

#include <dolfin/fem/UFC.h>

namespace dolfin
{

  class Function;
  class Data;

  /// This is a utility class used by Function to wrap callbacks from
  /// ufc::finite_element::evaluate_dof to ufc::function::evaluate.

  // FIXME: Remove this class

  class UFCFunction : public ufc::function
  {
  public:

    /// Create wrapper for given function
    UFCFunction(const Function& v, Data& data);

    /// Destructor
    ~UFCFunction();

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(double* values,
                  const double* coordinates,
                  const ufc::cell& cell) const;

  private:

    // The function
    const Function& v;

    // Function call data
    mutable Data& data;

  };

}

#endif
