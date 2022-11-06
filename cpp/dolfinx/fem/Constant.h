// Copyright (C) 2019-2021 Chris Richardson, Michal Habera and Garth N.
// Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

namespace dolfinx::fem
{

/// Constant value which can be attached to a Form. Constants may be
/// scalar (rank 0), vector (rank 1), or tensor-valued.
template <typename T>
class Constant
{
public:
  /// @brief Create a rank-0 (scalar-valued) constant
  /// @param[in] c Value of the constant
  explicit Constant(T c) : value({c})
  {
  }

  /// @brief Create a rank-1 (vector-valued) constant
  /// @param[in] c Value of the constant
  explicit Constant(std::span<const T> c)
      : Constant(c, std::vector<std::size_t>{c.size()})
  {
  }

  /// @brief Create a rank-d constant
  /// @param[in] c Value of the Constant (row-majors storage)
  /// @param[in] shape Shape of the Constant
  Constant(std::span<const T> c, std::span<const std::size_t> shape)
      : value(c.begin(), c.end()), shape(shape.begin(), shape.end())
  {
  }

  /// Values, stored as a row-major flattened array
  std::vector<T> value;

  /// Shape
  std::vector<std::size_t> shape;
};
} // namespace dolfinx::fem
