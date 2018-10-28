// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormCoefficients.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/log/log.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormCoefficients::FormCoefficients(const ufc_form& ufc_form)
    : _coefficients(ufc_form.num_coefficients)
{
  // Create finite elements for coefficients
  for (int i = 0; i < ufc_form.num_coefficients; i++)
  {
    std::shared_ptr<ufc_finite_element> element(
        ufc_form.create_finite_element(ufc_form.rank + i));

    _elements.push_back(fem::FiniteElement(element));
    _original_pos.push_back(ufc_form.original_coefficient_position(i));
  }
}
//-----------------------------------------------------------------------------
FormCoefficients::FormCoefficients(
    std::vector<fem::FiniteElement>& coefficient_elements)
    : _elements(coefficient_elements),
      _coefficients(coefficient_elements.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t FormCoefficients::size() const { return _coefficients.size(); }
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    std::size_t i, std::shared_ptr<const function::GenericFunction> coefficient)
{
  assert(i < _coefficients.size());
  _coefficients[i] = coefficient;

  // FIXME: if GenericFunction has an element, check it matches

  // Check value_rank and value_size of GenericFunction match those of
  // FiniteElement i.

  const std::size_t value_rank = coefficient->value_rank();
  const std::size_t value_rank_fe = _elements[i].value_rank();
  if (value_rank_fe != value_rank)
  {
    log::dolfin_error(
        "FormCoefficients.h", "set coefficient",
        "Invalid value rank for coefficient %d (got %d but expecting %d). "
        "You might have forgotten to specify the value rank correctly in an "
        "Expression subclass",
        i, value_rank, value_rank_fe);
  }

  for (std::size_t j = 0; j < value_rank; ++j)
  {
    const std::size_t dim = coefficient->value_dimension(j);
    const std::size_t dim_fe = _elements[i].value_dimension(j);
    if (dim != dim_fe)
    {
      log::dolfin_error(
          "FormCoefficients.h", "set coefficient",
          "Invalid value dimension %d for coefficient %d (got %d "
          "but expecting %d). "
          "You might have forgotten to specify the value dimension "
          "correctly in an Expression subclass ",
          j, i, dim, dim_fe);
    }
  }
}
//-----------------------------------------------------------------------------
const function::GenericFunction* FormCoefficients::get(std::size_t i) const
{
  assert(i < _coefficients.size());
  return _coefficients[i].get();
}
//-----------------------------------------------------------------------------
const fem::FiniteElement& FormCoefficients::element(std::size_t i) const
{
  assert(i < _elements.size());
  return _elements[i];
}
//-----------------------------------------------------------------------------
std::size_t FormCoefficients::original_position(std::size_t i) const
{
  assert(i < _original_pos.size());
  return _original_pos[i];
}
//-----------------------------------------------------------------------------
