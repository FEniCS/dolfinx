// Copyright (C) 2006-2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-08
// Last changed: 2008-05-08

#ifndef __UFC_FUNCTION_H
#define __UFC_FUNCTION_H

#include "GenericFunction.h"

namespace dolfin
{
  // TODO: Take rank and shape instead of size in constructor. Or better: add rank() and dim(i) to ufc::function in next version.

  /// This class implements the functionality for functions
  /// that take a single constant value.

  class UFCFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Create function to wrap given ufc::function 
    UFCFunction(Mesh& mesh, const ufc::function& function, uint size);

    /// Destructor
    ~UFCFunction();

    /// Return the rank of the value space
    uint rank() const;

    /// Return the dimension of the value space for axis i
    uint dim(uint i) const;

    /// Interpolate function to vertices of mesh
    void interpolate(real* values) const;

    /// Interpolate function to finite element space on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element) const;

    /// Evaluate function at given point
    void eval(real* values, const real* x) const;

    /// Evaluate function at given point in cell (UFC function interface)
    void evaluate(real* values,
                  const real* coordinates,
                  const ufc::cell& cell) const;

  private:

    // Underlying ufc::function
    const ufc::function & function;

    // Size of value (number of entries in tensor value)
    uint size;

  };

}

#endif
