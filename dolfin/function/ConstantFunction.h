// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-07-07


#ifndef __CONSTANT_FUNCTION_H
#define __CONSTANT_FUNCTION_H

#include "GenericFunction.h"

namespace dolfin
{

  /// This class implements the functionality for functions
  /// that take a single constant value.

  class ConstantFunction : public GenericFunction, public ufc::function
  {
  public:

    /// Copy constructor
    ConstantFunction(const ConstantFunction& f);

    /// Create constant scalar function from given value
    ConstantFunction(Mesh& mesh, real value);

    /// Create constant vector function from given size and value
    ConstantFunction(Mesh& mesh, uint size, real value);

    /// Create constant vector function from given size and values
    ConstantFunction(Mesh& mesh, const Array<real>& values);

    /// Create constant tensor function from given shape and values
    ConstantFunction(Mesh& mesh, const Array<uint>& shape, const Array<real>& values);

    /// Destructor
    ~ConstantFunction();

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

    // Values of constant function
    real* values;

    // Tensor rank
    uint value_rank;

    // Tensor shape
    uint* shape;

    // Size of value (number of entries in tensor value)
    uint size;

  };

}

#endif
