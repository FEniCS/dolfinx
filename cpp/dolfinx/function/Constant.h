// Copyright (C) 2019 Chris Richardson and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace dolfinx::function
{

/// A constant value which can be attached to a Form.
/// Constants may be scalar (rank 0), vector (rank 1), or tensor valued.
template <typename T>
class Constant
{

public:
  /// Create a rank-0 (scalar-valued) constant
  explicit Constant(T c) : value({c}) {}

  /// Create a rank-1 (vector-valued) constant
  explicit Constant(const std::vector<T>& c) : shape(1, c.size()), value({c}) {}

  /// Create a rank-2 constant
  explicit Constant(
      const Eigen::Ref<
          Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& c)
      : shape({(int)c.rows(), (int)c.cols()}), value(c.rows() * c.cols())
  {
    for (int i = 0; i < c.rows(); ++i)
      for (int j = 0; j < c.cols(); ++j)
        value[i * c.cols() + j] = c(i, j);
  }

  /// Create an arbitrary rank constant. Data layout is row-major (C style).
  Constant(std::vector<int> shape, std::vector<T> value)
      : shape(shape), value(value)
  {
    // Do nothing
  }

  /// Shape
  std::vector<int> shape;

  /// Values, stored as a flattened array.
  std::vector<T> value;
};
} // namespace dolfinx::function
