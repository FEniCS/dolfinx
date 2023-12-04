// Copyright (C) 2019-2023 Chris Richardson, Michal Habera and Garth N.
// Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/common/types.h"
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Constant value which can be attached to a Form.
///
/// Constants may be scalar (rank 0), vector (rank 1), or tensor-valued.
/// @tparam T Scalar type of the Constant.
template <dolfinx::scalar T>
class Constant
{
public:
  /// Field type
  using value_type = T;

  /// @brief Create a rank-0 (scalar-valued) constant
  /// @param[in] c Value of the constant
  explicit Constant(value_type c) : value({c}) {}

  /// @brief Create a rank-1 (vector-valued) constant
  /// @param[in] c Value of the constant
  explicit Constant(std::span<const value_type> c)
      : Constant(c, std::vector<std::size_t>{c.size()})
  {
  }

  /// @brief Create a rank-d constant
  /// @param[in] c Value of the Constant (row-majors storage)
  /// @param[in] shape Shape of the Constant
  Constant(std::span<const value_type> c, std::span<const std::size_t> shape)
      : value(c.begin(), c.end()), shape(shape.begin(), shape.end())
  {
  }

  /// Values, stored as a row-major flattened array
  std::vector<value_type> value;

  /// Shape
  std::vector<std::size_t> shape;
};
} // namespace dolfinx::fem
