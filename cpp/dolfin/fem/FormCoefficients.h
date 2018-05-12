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
class GenericFunction;
}

namespace fem
{
class FiniteElement;

/// Storage for the coefficients of a Form consisting of GenericFunctions and
/// the Elements they are defined on
class FormCoefficients
{
public:
  /// Initialise the FormCoefficients from a ufc_form, instantiating all the
  /// required elements
  FormCoefficients(const ufc_form& ufc_form);

  /// Initialise the FormCoefficients with their elements only
  FormCoefficients(std::vector<fem::FiniteElement>& coefficient_elements);

  /// Get number of coefficients
  std::size_t size() const;

  /// Set a coefficient to be a GenericFunction
  void set(std::size_t i,
           std::shared_ptr<const function::GenericFunction> coefficient);

  /// Get the GenericFunction coefficient i
  std::shared_ptr<const function::GenericFunction> get(std::size_t i) const;

  /// Get the element for coefficient i
  const fem::FiniteElement& element(std::size_t i) const;

  /// Original position of coefficient in UFL form
  const std::size_t original_position(std::size_t i) const;

private:
  // Finite elements for coefficients
  std::vector<fem::FiniteElement> _elements;

  // GenericFunctions for the coefficients
  std::vector<std::shared_ptr<const function::GenericFunction>> _coefficients;

  // Copy of 'original positions' in UFL form
  std::vector<std::size_t> _original_pos;
};
} // namespace fem
} // namespace dolfin
