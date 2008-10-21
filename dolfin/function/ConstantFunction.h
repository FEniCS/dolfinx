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

  class ConstantFunction : public Function, public ufc::function
  {
  public:

    /*
    /// Copy constructor
    ConstantFunction(const ConstantFunction& f);

    /// Create constant scalar function from given value
    ConstantFunction(Mesh& mesh, double value);

    /// Create constant vector function from given size and value
    ConstantFunction(Mesh& mesh, uint size, double value);

    /// Create constant vector function from given size and values
    ConstantFunction(Mesh& mesh, const Array<double>& values);

    /// Create constant tensor function from given shape and values
    ConstantFunction(Mesh& mesh, const Array<uint>& shape, const Array<double>& values);

    /// Destructor
    ~ConstantFunction();
    */

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
