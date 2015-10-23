// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug, 2009
// Modified by Garth N. Wells, 2010
// Modified by Martin Alnaes, 2013-2015

#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include "GenericDofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "UFC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const Form& a) : form(*a.ufc_form()), coefficients(a.coefficients()),
                          dolfin_form(a)
{
  dolfin_assert(a.ufc_form());
  init(a);
}
//-----------------------------------------------------------------------------
UFC::UFC(const UFC& ufc) : form(ufc.form),
                           coefficients(ufc.dolfin_form.coefficients()),
                           dolfin_form(ufc.dolfin_form)
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

  // Create finite elements for coefficients
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    std::shared_ptr<ufc::finite_element>
      element(form.create_finite_element(form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Create cell integrals
  default_cell_integral
    = std::shared_ptr<ufc::cell_integral>(form.create_default_cell_integral());
  for (std::size_t i = 0; i < form.max_cell_subdomain_id(); i++)
    cell_integrals.push_back(std::shared_ptr<ufc::cell_integral>(form.create_cell_integral(i)));

  // Create exterior facet integrals
  default_exterior_facet_integral
    = std::shared_ptr<ufc::exterior_facet_integral>(form.create_default_exterior_facet_integral());
  for (std::size_t i = 0; i < form.max_exterior_facet_subdomain_id(); i++)
    exterior_facet_integrals.push_back(std::shared_ptr<ufc::exterior_facet_integral>(form.create_exterior_facet_integral(i)));

  // Create interior facet integrals
  default_interior_facet_integral
    = std::shared_ptr<ufc::interior_facet_integral>(form.create_default_interior_facet_integral());
  for (std::size_t i = 0; i < form.max_interior_facet_subdomain_id(); i++)
    interior_facet_integrals.push_back(std::shared_ptr<ufc::interior_facet_integral>(form.create_interior_facet_integral(i)));

  // Create point integrals
  default_vertex_integral
    = std::shared_ptr<ufc::vertex_integral>(this->form.create_default_vertex_integral());
  for (std::size_t i = 0; i < this->form.max_vertex_subdomain_id(); i++)
    vertex_integrals.push_back(std::shared_ptr<ufc::vertex_integral>(this->form.create_vertex_integral(i)));

  // Create custom integrals
  default_custom_integral
    = std::shared_ptr<ufc::custom_integral>(this->form.create_default_custom_integral());
  for (std::size_t i = 0; i < this->form.max_custom_subdomain_id(); i++)
    custom_integrals.push_back(std::shared_ptr<ufc::custom_integral>(this->form.create_custom_integral(i)));

  // Create cutcell integrals
  default_cutcell_integral
    = std::shared_ptr<ufc::cutcell_integral>(this->form.create_default_cutcell_integral());
  for (std::size_t i = 0; i < this->form.max_cutcell_subdomain_id(); i++)
    cutcell_integrals.push_back(std::shared_ptr<ufc::cutcell_integral>(this->form.create_cutcell_integral(i)));

  // Create interface integrals
  default_interface_integral
    = std::shared_ptr<ufc::interface_integral>(this->form.create_default_interface_integral());
  for (std::size_t i = 0; i < this->form.max_interface_subdomain_id(); i++)
    interface_integrals.push_back(std::shared_ptr<ufc::interface_integral>(this->form.create_interface_integral(i)));

  // Create overlap integrals
  default_overlap_integral
    = std::shared_ptr<ufc::overlap_integral>(this->form.create_default_overlap_integral());
  for (std::size_t i = 0; i < this->form.max_overlap_subdomain_id(); i++)
    overlap_integrals.push_back(std::shared_ptr<ufc::overlap_integral>(this->form.create_overlap_integral(i)));

  // Get maximum local dimensions
  std::vector<std::size_t> max_element_dofs;
  std::vector<std::size_t> max_macro_element_dofs;
  for (std::size_t i = 0; i < form.rank(); i++)
  {
    dolfin_assert(V[i]->dofmap());
    max_element_dofs.push_back(V[i]->dofmap()->max_element_dofs());
    max_macro_element_dofs.push_back(2*V[i]->dofmap()->max_element_dofs());
  }

  // Initialize local tensor
  std::size_t num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_element_dofs[i];
  A.resize(num_entries);
  A_facet.resize(num_entries);

  // Initialize local tensor for macro element
  num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_macro_element_dofs[i];
  macro_A.resize(num_entries);

  // Initialize coefficients
  _w.resize(form.num_coefficients());
  w_pointer.resize(form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    _w[i].resize(coefficient_elements[i].space_dimension());
    w_pointer[i] = _w[i].data();
  }

  // Initialize coefficients on macro element
  _macro_w.resize(form.num_coefficients());
  macro_w_pointer.resize(form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    const std::size_t n = 2*coefficient_elements[i].space_dimension();
    _macro_w[i].resize(n);
    macro_w_pointer[i] = _macro_w[i].data();
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c, const std::vector<double>& coordinate_dofs,
                 const ufc::cell& ufc_cell,
                 const std::vector<bool> & enabled_coefficients)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    dolfin_assert(coefficients[i]);
    coefficients[i]->restrict(_w[i].data(), coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c0, const std::vector<double>& coordinate_dofs0,
                 const ufc::cell& ufc_cell0,
                 const Cell& c1, const std::vector<double>& coordinate_dofs1,
                 const ufc::cell& ufc_cell1,
                 const std::vector<bool> & enabled_coefficients)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    dolfin_assert(coefficients[i]);
    const std::size_t offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(_macro_w[i].data(), coefficient_elements[i],
                              c0, coordinate_dofs0.data(), ufc_cell0);
    coefficients[i]->restrict(_macro_w[i].data() + offset,
                              coefficient_elements[i],
                              c1, coordinate_dofs1.data(), ufc_cell1);
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
    coefficients[i]->restrict(_w[i].data(), coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c0, const std::vector<double>& coordinate_dofs0,
                 const ufc::cell& ufc_cell0,
                 const Cell& c1, const std::vector<double>& coordinate_dofs1,
                 const ufc::cell& ufc_cell1)
{
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    dolfin_assert(coefficients[i]);
    const std::size_t offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(_macro_w[i].data(), coefficient_elements[i],
                              c0, coordinate_dofs0.data(), ufc_cell0);
    coefficients[i]->restrict(_macro_w[i].data() + offset,
                              coefficient_elements[i],
                              c1, coordinate_dofs1.data(), ufc_cell1);
  }
}
//-----------------------------------------------------------------------------
