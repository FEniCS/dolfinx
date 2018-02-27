// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UFC.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/Cell.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const Form& a) : dolfin_form(a)
{
  // Get function spaces for arguments
  std::vector<std::shared_ptr<const FunctionSpace>> V = a.function_spaces();

  //
  // Initialise temporary space for element tensors
  //

  // FIXME: make Assembler responsible for this

  // Get maximum local dimensions
  const std::size_t num_entries = a.max_element_tensor_size();
  A.resize(num_entries);
  macro_A.resize(num_entries * std::pow(2, a.rank()));

  //
  // Initialize storage for coefficient values
  //
  std::size_t num_coeffs = a.coeffs().size();

  // Calculate size and offsets for coefficient values
  std::vector<std::size_t> n = {0};
  for (std::size_t i = 0; i < num_coeffs; ++i)
  {
    const auto& element = a.coeffs().element(i);
    n.push_back(n.back() + element.space_dimension());
  }

  _w.resize(n.back());
  _macro_w.resize(2 * n.back());

  w_pointer.resize(num_coeffs);
  macro_w_pointer.resize(num_coeffs);
  for (std::size_t i = 0; i < num_coeffs; ++i)
  {
    w_pointer[i] = _w.data() + n[i];
    macro_w_pointer[i] = _macro_w.data() + 2 * n[i];
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const mesh::Cell& c,
                 Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs,
                 const ufc::cell& ufc_cell,
                 const std::vector<bool>& enabled_coefficients)
{
  const auto& coefficients = dolfin_form.coeffs();

  // Restrict coefficients to cell
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    const auto coefficient = coefficients.get(i);
    const auto& element = coefficients.element(i);
    coefficient->restrict(w_pointer[i], element, c, coordinate_dofs.data(),
                          ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const mesh::Cell& c,
                 const std::vector<double>& coordinate_dofs,
                 const ufc::cell& ufc_cell,
                 const std::vector<bool>& enabled_coefficients)
{
  const auto& coefficients = dolfin_form.coeffs();

  // Restrict coefficients to cell
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    const auto coefficient = coefficients.get(i);
    const auto& element = coefficients.element(i);
    coefficient->restrict(w_pointer[i], element, c, coordinate_dofs.data(),
                          ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const mesh::Cell& c0,
                 const std::vector<double>& coordinate_dofs0,
                 const ufc::cell& ufc_cell0, const mesh::Cell& c1,
                 const std::vector<double>& coordinate_dofs1,
                 const ufc::cell& ufc_cell1,
                 const std::vector<bool>& enabled_coefficients)
{
  const auto& coefficients = dolfin_form.coeffs();

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    const auto coefficient = coefficients.get(i);
    const auto& element = coefficients.element(i);
    const std::size_t offset = element.space_dimension();
    coefficient->restrict(macro_w_pointer[i], element, c0,
                          coordinate_dofs0.data(), ufc_cell0);
    coefficient->restrict(macro_w_pointer[i] + offset, element, c1,
                          coordinate_dofs1.data(), ufc_cell1);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const mesh::Cell& c0,
                 Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs0,
                 const ufc::cell& ufc_cell0, const mesh::Cell& c1,
                 Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs1,
                 const ufc::cell& ufc_cell1,
                 const std::vector<bool>& enabled_coefficients)
{
  const auto& coefficients = dolfin_form.coeffs();

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    const auto coefficient = coefficients.get(i);
    const auto& element = coefficients.element(i);
    const std::size_t offset = element.space_dimension();
    coefficient->restrict(macro_w_pointer[i], element, c0,
                          coordinate_dofs0.data(), ufc_cell0);
    coefficient->restrict(macro_w_pointer[i] + offset, element, c1,
                          coordinate_dofs1.data(), ufc_cell1);
  }
}
//-----------------------------------------------------------------------------
