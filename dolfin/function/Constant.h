// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008-2011.

#ifndef __CONSTANT_H
#define __CONSTANT_H

#include <vector>
#include "Expression.h"

namespace dolfin
{
  template<typename T> class Array;
  class Mesh;

  /// This class represents a constant-valued expression.

  class Constant : public Expression
  {
  public:

    /// Create scalar constant
    ///
    /// *Arguments*
    ///     value (double)
    ///         The scalar to create a Constant object from.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Constant c(1.0);
    explicit Constant(double value);

    /// Create vector constant (dim = 2)
    ///
    /// *Arguments*
    ///     value0 (double)
    ///         The first vector element.
    ///     value1 (double)
    ///         The second vector element.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Constant B(0.0, 1.0);
    Constant(double value0, double value1);

    /// Create vector constant (dim = 3)
    ///
    /// *Arguments*
    ///     value0 (double)
    ///         The first vector element.
    ///     value1 (double)
    ///         The second vector element.
    ///     value2 (double)
    ///         The third vector element.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Constant T(0.0, 1.0, 0.0);
    Constant(double value0, double value1, double value2);

    /// Create vector-valued constant
    ///
    /// *Arguments*
    ///     values (std::vector<double>)
    ///         Values to create a vector-valued constant from.
    explicit Constant(std::vector<double> values);

    /// Create tensor-valued constant for flattened array of values
    ///
    /// *Arguments*
    ///     value_shape (std::vector<std::size_t>)
    ///         Shape of tensor.
    ///     values (std::vector<double>)
    ///         Values to create tensor-valued constant from.
    Constant(std::vector<std::size_t> value_shape,
             std::vector<double> values);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     constant (_Constant_)
    ///         Object to be copied.
    Constant(const Constant& constant);

    /// Destructor
    ~Constant();

    /// Assignment operator
    ///
    /// *Arguments*
    ///     constant (_Constant_)
    ///         Another constant.
    const Constant& operator= (const Constant& constant);

    /// Assignment operator
    ///
    /// *Arguments*
    ///     constant (double)
    ///         Another constant.
    const Constant& operator= (double constant);

    /// Cast to double (for scalar constants)
    ///
    /// *Returns*
    ///     double
    ///         The scalar value.
    operator double() const;

    /// Return copy of this Constant's current values
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         The vector of scalar values of the constant.
    std::vector<double> values() const;

    //--- Implementation of Expression interface ---

    void eval(Array<double>& values, const Array<double>& x) const;

    virtual std::string str(bool verbose) const;

  private:

    // Values of constant function
    std::vector<double> _values;

  };

}

#endif
