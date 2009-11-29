// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2009-10-05

#ifndef __CONSTANT_H
#define __CONSTANT_H

#include <vector>
#include "Expression.h"

namespace dolfin
{

  class Mesh;

  /// This class represents a constant-valued expression.

  class Constant : public Expression
  {
  public:

    /// Create scalar constant
    Constant(double value);

    /// Create vector constant (dim = 2)
    Constant(double value0, double value1);

    /// Create vector constant (dim = 3)
    Constant(double value0, double value1, double value2);

    /// Create vector-valued constant
    Constant(std::vector<double> values);

    /// Create tensor-valued constant for flattened array of values
    Constant(std::vector<uint> value_shape,
             std::vector<double> values);

    /// Copy constructor
    Constant(const Constant& constant);

    /// Destructor
    ~Constant();

    /// Assignment operator
    const Constant& operator= (const Constant& constant);

    /// Assignment operator
    const Constant& operator= (double constant);

    /// Cast to double (for scalar constants)
    operator double() const;

    //--- Implementation of Expression interface ---

    /// Evaluate function
    virtual void eval(std::vector<double>& values, const Data& data) const;

  private:

    // Values of constant function
    std::vector<double> _values;

  };

}

#endif
