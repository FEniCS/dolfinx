// Copyright (C) 2019 Chris Richardson and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>
#include <xtensor/xarray.hpp>

namespace dolfinx::fem
{

/// A constant value which can be attached to a Form.
/// Constants may be scalar (rank 0), vector (rank 1), or tensor valued.
template <typename T>
class Constant
{

public:
  /// Create a rank-0 (scalar-valued) constant
  explicit Constant(T c) : value({c}) {}

  /// Create a rank-d constant
  explicit Constant(const xt::xarray<T>& c)
      : value(c.data(), c.data() + c.size())
  {
    std::copy(c.shape().cbegin(), c.shape().cend(), std::back_inserter(shape));
  }

  /// Shape
  std::vector<int> shape;

  /// Values, stored as a flattened array.
  std::vector<T> value;
};
} // namespace dolfinx::fem
