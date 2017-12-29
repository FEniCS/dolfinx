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

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "Expression.h"

namespace dolfin
{
  class Mesh;

  /// This class represents a constant-valued expression.

  class Constant : public Expression
  {
  public:

    // FIXME: remove once Expression constructor is fixed for scalars
    /// Create scalar constant
    ///
    /// @param  value (double)
    ///         The scalar to create a Constant object from.
    ///
    /// @code{.cpp}
    ///         Constant c(1.0);
    /// @endcode
    explicit Constant(double value);

    /// Create vector-valued constant
    ///
    /// @param values (std::vector<double>)
    ///         Values to create a vector-valued constant from.
    explicit Constant(std::vector<double> values);

    /// Create tensor-valued constant for flattened array of values
    ///
    /// @param value_shape (std::vector<std::size_t>)
    ///         Shape of tensor.
    /// @param values (std::vector<double>)
    ///         Values to create tensor-valued constant from.
    Constant(std::vector<std::size_t> value_shape,
             std::vector<double> values);

    /// Copy constructor
    ///
    /// @param constant (Constant)
    ///         Object to be copied.
    Constant(const Constant& constant);

    /// Destructor
    ~Constant();

    /// Assignment operator
    ///
    /// @param constant (Constant)
    ///         Another constant.
    const Constant& operator= (const Constant& constant);

    /// Assignment operator
    ///
    /// @param constant (double)
    ///         Another constant.
    const Constant& operator= (double constant);

    /// Cast to double (for scalar constants)
    ///
    /// @return double
    ///         The scalar value.
    operator double() const;

    /// Return copy of this Constant's current values
    ///
    /// @return std::vector<double>
    ///         The vector of scalar values of the constant.
    std::vector<double> values() const;

    //--- Implementation of Expression interface ---

    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x) const override;

    virtual std::string str(bool verbose) const override;

  private:

    // Values of constant function
    std::vector<double> _values;

  };

}
