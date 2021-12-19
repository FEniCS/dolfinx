// Copyright (C) 2019-2021 Chris Richardson, Michal Habera and Garth N.
// Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>
#include <xtensor/xarray.hpp>

namespace dolfinx::fem
{

/// Constant value which can be attached to a Form. Constants may be
/// scalar (rank 0), vector (rank 1), or tensor valued.
template <typename T>
class Constant
{
public:
  /// Create a rank-0 (scalar-valued) constant
  /// @param[in] c Value of the constant
  explicit Constant(T c) : value({c}) {}

  /// Create a rank-d constant
  /// @param[in] c Value of the constant
  explicit Constant(const xt::xarray<T>& c)
      : shape(c.shape().begin(), c.shape().end()), value(c.begin(), c.end())
  {
  }

  /// Shape
  std::vector<int> shape;

  /// Values, stored as a row-major flattened array
  std::vector<T> value;
};
} // namespace dolfinx::fem
