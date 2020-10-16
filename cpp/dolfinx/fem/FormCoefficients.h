// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/function/Function.h>
#include <memory>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Function;
}

namespace fem
{
class FiniteElement;

/// Storage for the coefficients of a Form consisting of Function and
/// the Element objects they are defined on.

template <typename T>
class FormCoefficients
{
public:
  /// Initialise the FormCoefficients
  FormCoefficients(
      const std::vector<std::shared_ptr<const function::Function<T>>>&
          coefficients)
      : _coefficients(coefficients)
  {
    // Do nothing
  }

  /// Get number of coefficients
  int size() const { return _coefficients.size(); }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> offsets() const
  {
    std::vector<int> n{0};
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Get the Function coefficient i
  std::shared_ptr<const function::Function<T>> get(int i) const
  {
    return _coefficients.at(i);
  }

private:
  // Functions for the coefficients
  std::vector<std::shared_ptr<const function::Function<T>>> _coefficients;
};
} // namespace fem
} // namespace dolfinx
