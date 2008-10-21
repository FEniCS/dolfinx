// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-07-07

#ifndef __CONSTANT_FUNCTION_H
#define __CONSTANT_FUNCTION_H

#include "Function.h"

namespace dolfin
{

  /// This class implements the functionality for functions
  /// that take a single constant value.

  class ConstantFunction : public Function
  {
  public:

    /// Copy constructor
    //ConstantFunction(const ConstantFunction& f);

    /// Create constant scalar function from given value
    explicit ConstantFunction(double value);

    /// Create constant vector function from given size and value
    ConstantFunction(uint size, double value);

    /// Create constant vector function from given size and values
    ConstantFunction(const Array<double>& values);

    /// Create constant tensor function from given shape and values
    ConstantFunction(const Array<uint>& shape, const Array<double>& values);

    /// Destructor
    ~ConstantFunction();

    // FIXME: Why are the interpolate functions needed? Shouldn't it be
    // FIXME: enough to implement eval()?

    /// Interpolate function to vertices of mesh
    void interpolate(double* values, const FunctionSpace& V) const;

    /// Interpolate function to finite element space on cell
    void interpolate(double* coefficients,
                     const ufc::cell& cell,
                     const FunctionSpace& V) const;

    /// Evaluate function at given point
    void eval(double* values, const double* x) const;

  private:

    // Values of constant function
    double* values;

    // Tensor rank
    uint value_rank;

    // Tensor shape
    uint* shape;

    // Size of value (number of entries in tensor value)
    uint size;

  };

}

#endif
