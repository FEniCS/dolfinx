// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-10-28

#ifndef __CONSTANT_H
#define __CONSTANT_H

#include "Function.h"

namespace dolfin
{

  /// This class implements the functionality for functions
  /// that take a single constant value.

  class Constant : public Function
  {
  public:

    /// Copy constructor
    //Constant(const ConstantFunction& f);

    /// Create constant scalar function from given value
    explicit Constant(double value);

    /*
    /// Create constant vector function from given size and value
    Constant(uint size, double value);

    /// Create constant vector function from given size and values
    Constant(const Array<double>& values);

    /// Create constant tensor function from given shape and values
    Constant(const Array<uint>& shape, const Array<double>& values);

    */

    /// Destructor
    ~Constant();

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
