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

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const Form& a) : coefficients(a.coefficients()), dolfin_form(a)
{
  dolfin_assert(a.ufc_form());

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
  std::size_t num_coeffs = a.num_coefficients();

  // Create finite elements for coefficients
  for (std::size_t i = 0; i < num_coeffs; i++)
  {
    std::shared_ptr<ufc::finite_element> element(
        a.ufc_form()->create_finite_element(a.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Calculate size and offsets for coefficient values
  std::vector<std::size_t> n = {0};
  for (std::size_t i = 0; i < num_coeffs; ++i)
    n.push_back(n.back() + coefficient_elements[i].space_dimension());
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
void UFC::update(const Cell& c,
                 Eigen::Ref<const Eigen::Matrix<
                     double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                     coordinate_dofs,
                 const ufc::cell& ufc_cell,
                 const std::vector<bool>& enabled_coefficients)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    dolfin_assert(coefficients[i]);
    coefficients[i]->restrict(w_pointer[i], coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c, const std::vector<double>& coordinate_dofs,
                 const ufc::cell& ufc_cell,
                 const std::vector<bool>& enabled_coefficients)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    dolfin_assert(coefficients[i]);
    coefficients[i]->restrict(w_pointer[i], coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c0, const std::vector<double>& coordinate_dofs0,
                 const ufc::cell& ufc_cell0, const Cell& c1,
                 const std::vector<double>& coordinate_dofs1,
                 const ufc::cell& ufc_cell1,
                 const std::vector<bool>& enabled_coefficients)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    dolfin_assert(coefficients[i]);
    const std::size_t offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(macro_w_pointer[i], coefficient_elements[i], c0,
                              coordinate_dofs0.data(), ufc_cell0);
    coefficients[i]->restrict(macro_w_pointer[i] + offset,
                              coefficient_elements[i], c1,
                              coordinate_dofs1.data(), ufc_cell1);
  }
}
//-----------------------------------------------------------------------------
