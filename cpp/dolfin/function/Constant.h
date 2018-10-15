// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include <Eigen/Dense>
#include <petscsys.h>
#include <vector>

namespace dolfin
{
class Mesh;

namespace function
{

/// This class represents a constant-valued expression.

class Constant : public Expression
{
public:
  // FIXME: remove once Expression constructor is fixed for scalars
  /// Create scalar constant
  ///
  /// @param  value (PetscScalar)
  ///         The scalar to create a Constant object from.
  ///
  /// @code{.cpp}
  ///         Constant c(1.0);
  ///         Constant c(1.0,1.0);
  /// @endcode
  explicit Constant(PetscScalar value);

  /// Create vector-valued constant
  ///
  /// @param values (std::vector<PetscScalar>)
  ///         Values to create a vector-valued constant from.
  explicit Constant(std::vector<PetscScalar> values);

  /// Create tensor-valued constant for flattened array of values
  ///
  /// @param value_shape (std::vector<std::size_t>)
  ///         Shape of tensor.
  /// @param values (std::vector<PetscScalar>)
  ///         Values to create tensor-valued constant from.
  Constant(std::vector<std::size_t> value_shape,
           std::vector<PetscScalar> values);

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
  const Constant& operator=(const Constant& constant);

  /// Assignment operator
  ///
  /// @param constant (PetscScalar)
  ///         Another constant.
  const Constant& operator=(PetscScalar constant);

  /// Return copy of this Constant's current values
  ///
  /// @return std::vector<PetscScalar>
  ///         The vector of scalar values of the constant.
  std::vector<PetscScalar> values() const;

  //--- Implementation of Expression interface ---

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            const Eigen::Ref<const EigenRowArrayXXd> x) const override;

  virtual std::string str(bool verbose) const override;

private:
  // Values of constant function
  std::vector<PetscScalar> _values;
};
} // namespace function
} // namespace dolfin
