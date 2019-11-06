// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

namespace function
{
class Function;
}

namespace fem
{
class FiniteElement;

/// Storage for the coefficients of a Form consisting of Function and
/// the Element objects they are defined on.

class FormCoefficients
{
public:
  /// Initialise the FormCoefficients, using tuples of
  /// (original_coeff_position, name, shared_ptr<function::Function>).
  /// The shared_ptr<Function> may be a nullptr and assigned later.
  FormCoefficients(
      const std::vector<
          std::tuple<int, std::string, std::shared_ptr<function::Function>>>&
          coefficients);

  /// Get number of coefficients
  int size() const;

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> offsets() const;

  /// Set coefficient with index i to be a Function
  void set(int i, std::shared_ptr<const function::Function> coefficient);

  /// Set coefficient with name to be a Function
  void set(std::string name,
           std::shared_ptr<const function::Function> coefficient);

  /// Get the Function coefficient i
  std::shared_ptr<const function::Function> get(int i) const;

  /// Original position of coefficient in UFL form
  int original_position(int i) const;

  /// Get index from name of coefficient
  /// @param[in] name Name of coefficient
  /// @return Index of the coefficient
  int get_index(std::string name) const;

  /// Get name from index of coefficient
  /// @param[in] index Index of the coefficient
  /// @return Name of the coefficient
  std::string get_name(int index) const;

private:
  // Functions for the coefficients
  std::vector<std::shared_ptr<const function::Function>> _coefficients;

  // Copy of 'original positions' in UFL form
  std::vector<int> _original_pos;

  // Names of coefficients
  std::vector<std::string> _names;
};
} // namespace fem
} // namespace dolfin
