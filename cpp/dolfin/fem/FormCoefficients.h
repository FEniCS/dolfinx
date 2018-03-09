// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FiniteElement.h"
#include <dolfin/function/GenericFunction.h>
#include <dolfin/log/log.h>
#include <functional>
#include <iostream>
#include <memory>
#include <ufc.h>
#include <vector>

namespace dolfin
{

namespace function
{
class GenericFunction;
}

namespace fem
{

/// Storage for the coefficients of a Form consisting
/// of GenericFunctions and the Elements they are defined on
class FormCoefficients
{
public:
  /// Initialise the FormCoefficients from a ufc::form
  /// instantiating all the required elements
  FormCoefficients(const ufc::form& ufc_form)
      : _coefficients(ufc_form.num_coefficients())
  {
    // Create finite elements for coefficients
    for (std::size_t i = 0; i < ufc_form.num_coefficients(); i++)
    {
      std::shared_ptr<ufc::finite_element> element(
          ufc_form.create_finite_element(ufc_form.rank() + i));
      _elements.push_back(fem::FiniteElement(element));
      _original_pos.push_back(ufc_form.original_coefficient_position(i));
    }
  }

  /// Get number of coefficients
  std::size_t size() const { return _coefficients.size(); }

  /// Set a coefficient to be a GenericFunction
  void set(std::size_t i,
           std::shared_ptr<const function::GenericFunction> coefficient)
  {
    dolfin_assert(i < _coefficients.size());
    _coefficients[i] = coefficient;

    // FIXME: if GenericFunction has an element, check it matches

    // Check value_rank and value_size of GenericFunction match those of
    // FiniteElement i.

    const std::size_t r = coefficient->value_rank();
    const std::size_t fe_r = _elements[i].value_rank();
    if (fe_r != r)
    {
      log::dolfin_error(
          "FormCoefficients.h", "set coefficient",
          "Invalid value rank for coefficient %d (got %d but expecting %d). "
          "You might have forgotten to specify the value rank correctly in an "
          "Expression subclass",
          i, r, fe_r);
    }

    for (std::size_t j = 0; j < r; ++j)
    {
      const std::size_t dim = coefficient->value_dimension(j);
      const std::size_t fe_dim = _elements[i].value_dimension(j);
      if (dim != fe_dim)
      {
        log::dolfin_error("FormCoefficients.h", "set coefficient",
                     "Invalid value dimension %d for coefficient %d (got %d "
                     "but expecting %d). "
                     "You might have forgotten to specify the value dimension "
                     "correctly in an Expression subclass ",
                     j, i, dim, fe_dim);
      }
    }
  }

  /// Get the GenericFunction coefficient i
  std::shared_ptr<const function::GenericFunction> get(std::size_t i) const
  {
    dolfin_assert(i < _coefficients.size());
    return _coefficients[i];
  }

  /// Get the element for coefficient i
  const fem::FiniteElement& element(std::size_t i) const
  {
    dolfin_assert(i < _elements.size());
    return _elements[i];
  }

  /// Original position of coefficient in UFL form
  const std::size_t original_position(std::size_t i) const
  {
    dolfin_assert(i < _original_pos.size());
    return _original_pos[i];
  }

private:
  // Finite elements for coefficients
  std::vector<fem::FiniteElement> _elements;

  // GenericFunctions for the coefficients
  std::vector<std::shared_ptr<const function::GenericFunction>> _coefficients;

  // Copy of 'original positions' in UFL form
  std::vector<std::size_t> _original_pos;
};
}
}
