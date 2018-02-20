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
  init(a);
}
//-----------------------------------------------------------------------------
UFC::UFC(const UFC& ufc)
    : coefficients(ufc.dolfin_form.coefficients()), dolfin_form(ufc.dolfin_form)
{
  this->init(ufc.dolfin_form);
}
//-----------------------------------------------------------------------------
UFC::~UFC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void UFC::init(const Form& a)
{
  // Get function spaces for arguments
  std::vector<std::shared_ptr<const FunctionSpace>> V = a.function_spaces();

  const ufc::form& form = *a.ufc_form();

  // Create finite elements for coefficients
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    std::shared_ptr<ufc::finite_element> element(
        form.create_finite_element(form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Create cell integrals
  default_cell_integral = std::shared_ptr<ufc::cell_integral>(
      form.create_default_cell_integral());
  for (std::size_t i = 0; i < form.max_cell_subdomain_id(); i++)
    cell_integrals.push_back(
        std::shared_ptr<ufc::cell_integral>(form.create_cell_integral(i)));

  // Create exterior facet integrals
  default_exterior_facet_integral
      = std::shared_ptr<ufc::exterior_facet_integral>(
          form.create_default_exterior_facet_integral());
  for (std::size_t i = 0; i < form.max_exterior_facet_subdomain_id(); i++)
    exterior_facet_integrals.push_back(
        std::shared_ptr<ufc::exterior_facet_integral>(
            form.create_exterior_facet_integral(i)));

  // Create interior facet integrals
  default_interior_facet_integral
      = std::shared_ptr<ufc::interior_facet_integral>(
          form.create_default_interior_facet_integral());
  for (std::size_t i = 0; i < form.max_interior_facet_subdomain_id(); i++)
    interior_facet_integrals.push_back(
        std::shared_ptr<ufc::interior_facet_integral>(
            form.create_interior_facet_integral(i)));

  // Create point integrals
  default_vertex_integral = std::shared_ptr<ufc::vertex_integral>(
      form.create_default_vertex_integral());
  for (std::size_t i = 0; i < form.max_vertex_subdomain_id(); i++)
    vertex_integrals.push_back(
        std::shared_ptr<ufc::vertex_integral>(form.create_vertex_integral(i)));

  // Create custom integrals
  default_custom_integral = std::shared_ptr<ufc::custom_integral>(
      form.create_default_custom_integral());
  for (std::size_t i = 0; i < form.max_custom_subdomain_id(); i++)
    custom_integrals.push_back(
        std::shared_ptr<ufc::custom_integral>(form.create_custom_integral(i)));

  // Get maximum local dimensions
  std::vector<std::size_t> max_element_dofs;
  std::vector<std::size_t> max_macro_element_dofs;
  for (std::size_t i = 0; i < form.rank(); i++)
  {
    dolfin_assert(V[i]->dofmap());
    max_element_dofs.push_back(V[i]->dofmap()->max_element_dofs());
    max_macro_element_dofs.push_back(2 * V[i]->dofmap()->max_element_dofs());
  }

  // Initialize local tensor
  std::size_t num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_element_dofs[i];
  A.resize(num_entries);

  // Initialize local tensor for macro element
  num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_macro_element_dofs[i];
  macro_A.resize(num_entries);

  // Initialize coefficients
  std::size_t num_coeffs = form.num_coefficients();

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
void UFC::update(const Cell& c, const std::vector<double>& coordinate_dofs,
                 const ufc::cell& ufc_cell)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    dolfin_assert(coefficients[i]);
    coefficients[i]->restrict(w_pointer[i], coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c0, const std::vector<double>& coordinate_dofs0,
                 const ufc::cell& ufc_cell0, const Cell& c1,
                 const std::vector<double>& coordinate_dofs1,
                 const ufc::cell& ufc_cell1)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
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
