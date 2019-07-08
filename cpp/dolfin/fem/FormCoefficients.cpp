// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormCoefficients.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <memory>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormCoefficients::FormCoefficients(
    const std::vector<std::tuple<int, std::string,
                                 std::shared_ptr<function::Function>>>& coeffs)
{
  for (const auto& coeff : coeffs)
  {
    _original_pos.push_back(std::get<0>(coeff));
    _names.push_back(std::get<1>(coeff));
    _coefficients.push_back(std::get<2>(coeff));
  }
}
//-----------------------------------------------------------------------------
int FormCoefficients::size() const { return _coefficients.size(); }
//-----------------------------------------------------------------------------
std::vector<int> FormCoefficients::offsets() const
{
  std::vector<int> n = {0};
  for (auto& c : _coefficients)
  {
    if (!c)
      throw std::runtime_error("Not all form coefficients have been set.");
    n.push_back(n.back() + c->function_space()->element->space_dimension());
  }
  return n;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    int i, std::shared_ptr<const function::Function> coefficient)
{
  if (i >= (int)_coefficients.size())
    _coefficients.resize(i + 1);

  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    std::string name, std::shared_ptr<const function::Function> coefficient)
{
  int i = get_index(name);
  if (i >= (int)_coefficients.size())
    _coefficients.resize(i + 1);

  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::Function> FormCoefficients::get(int i) const
{
  assert(i < (int)_coefficients.size());
  return _coefficients[i];
}
//-----------------------------------------------------------------------------
int FormCoefficients::original_position(int i) const
{
  assert(i < (int)_original_pos.size());
  return _original_pos[i];
}
//-----------------------------------------------------------------------------
int FormCoefficients::get_index(std::string name) const
{
  auto it = std::find(_names.begin(), _names.end(), name);
  if (it == _names.end())
    throw std::runtime_error("Cannot find coefficient name:" + name);

  return std::distance(_names.begin(), it);
}
//-----------------------------------------------------------------------------
std::string FormCoefficients::get_name(int i) const
{
  if (i >= (int)_names.size())
    throw std::runtime_error("Invalid coefficient index");

  return _names[i];
}
//-----------------------------------------------------------------------------
