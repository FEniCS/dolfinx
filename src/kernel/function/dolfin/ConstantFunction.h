// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2007-04-12

#ifndef __CONSTANT_FUNCTION_H
#define __CONSTANT_FUNCTION_H

#include <dolfin/GenericFunction.h>

namespace dolfin
{

  /// This class implements the functionality for functions
  /// that take a single constant value.

  class ConstantFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Create constant function from given value
    ConstantFunction(Mesh& mesh, real value);

    /// Destructor
    ~ConstantFunction();

    /// Return the rank of the value space
    uint rank() const;

    /// Return the dimension of the value space for axis i
    uint dim(uint i) const;

    /// Interpolate function to vertices of mesh
    void interpolate(real* values);

    /// Interpolate function to finite element space on cell
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

    // Size of value (number of entries in tensor value)
    uint size;

  };

}

#endif
