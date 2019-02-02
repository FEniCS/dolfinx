// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <vector>

struct ufc_form;

namespace dolfin
{

namespace function
{
class Function;
}

namespace fem
{
class FiniteElement;

/// Holder for the coefficients of a Form.

class FormCoefficients
{
public:
  /// Initialise the FormCoefficients from a ufc_form, instantiating all
  /// the required elements
  FormCoefficients(const ufc_form& ufc_form);

  /// Get number of coefficients
  std::size_t size() const;

  /// Set a coefficient to be a Function
  void set(std::size_t i,
           std::shared_ptr<const function::Function> coefficient);

  /// Get the Function coefficient i
  std::shared_ptr<const function::Function> get(std::size_t i) const;

  /// Return the original position of coefficient in UFL form
  std::size_t original_position(std::size_t i) const;

private:
  // Functions for the coefficients
  std::vector<std::shared_ptr<const function::Function>> _coefficients;

  // 'Original positions' in UFL form
  std::vector<std::size_t> _original_pos;
};
} // namespace fem
} // namespace dolfin
