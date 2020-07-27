// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <dolfinx/function/Function.h>
#include <memory>
#include <string>
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
  /// Initialise the FormCoefficients, using tuples of
  /// (original_coeff_position, name, Function). The Function pointer
  /// may be a nullptr and assigned later.
  FormCoefficients(
      const std::vector<
          std::tuple<int, std::string, std::shared_ptr<function::Function<T>>>>&
          coefficients)
  {
    for (const auto& c : coefficients)
    {
      _original_pos.push_back(std::get<0>(c));
      _names.push_back(std::get<1>(c));
      _coefficients.push_back(std::get<2>(c));
    }
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

  /// Set coefficient with index i to be a Function
  void set(int i,
           const std::shared_ptr<const function::Function<T>>& coefficient)
  {
    if (i >= (int)_coefficients.size())
      _coefficients.resize(i + 1);
    _coefficients[i] = coefficient;
  }

  /// Set coefficient with name to be a Function
  void set(const std::string& name,
           const std::shared_ptr<const function::Function<T>>& coefficient)
  {
    const int i = get_index(name);
    if (i >= (int)_coefficients.size())
      _coefficients.resize(i + 1);
    _coefficients[i] = coefficient;
  }

  /// Get the Function coefficient i
  std::shared_ptr<const function::Function<T>> get(int i) const
  {
    return _coefficients.at(i);
  }

  /// Original position of coefficient in UFL form
  /// @return The position of coefficient i in original ufl form
  ///   coefficients.
  int original_position(int i) const { return _original_pos.at(i); }

  /// Get index from name of coefficient
  /// @param[in] name Name of coefficient
  /// @return Index of the coefficient
  int get_index(const std::string& name) const
  {
    auto it = std::find(_names.begin(), _names.end(), name);
    if (it == _names.end())
      throw std::runtime_error("Cannot find coefficient name:" + name);
    return std::distance(_names.begin(), it);
  }

  /// Get name from index of coefficient
  /// @param[in] index Index of the coefficient
  /// @return Name of the coefficient
  std::string get_name(int index) const
  {
    if (index >= (int)_names.size())
      throw std::runtime_error("Invalid coefficient index");
    return _names[index];
  }

private:
  // Functions for the coefficients
  std::vector<std::shared_ptr<const function::Function<T>>> _coefficients;

  // Copy of 'original positions' in UFL form
  std::vector<int> _original_pos;

  // Names of coefficients
  std::vector<std::string> _names;
};
} // namespace fem
} // namespace dolfinx
