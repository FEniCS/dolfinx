// Copyright (C) 2007-2008 Anders Logg
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
//
// First added:  2007-01-17
// Last changed: 2011-02-21

#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include "GenericDofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "UFC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const Form& a) : form(*a.ufc_form()), cell(a.mesh()),
  cell0(a.mesh()), cell1(a.mesh()),
  coefficients(a.coefficients()), dolfin_form(a)
{
  dolfin_assert(a.ufc_form());
  init(a);
}
//-----------------------------------------------------------------------------
UFC::UFC(const UFC& ufc) : form(ufc.form), cell(ufc.dolfin_form.mesh()),
   cell0(ufc.dolfin_form.mesh()), cell1(ufc.dolfin_form.mesh()),
   coefficients(ufc.dolfin_form.coefficients()), dolfin_form(ufc.dolfin_form)
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
  std::vector<boost::shared_ptr<const FunctionSpace> > V = a.function_spaces();

  // Create finite elements for coefficients
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    boost::shared_ptr<ufc::finite_element> element(form.create_finite_element(form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Create cell integrals
  default_cell_integral = boost::shared_ptr<ufc::cell_integral>(form.create_default_cell_integral());
  for (std::size_t i = 0; i < form.num_cell_domains(); i++)
    cell_integrals.push_back(boost::shared_ptr<ufc::cell_integral>(form.create_cell_integral(i)));

  // Create exterior facet integrals
  default_exterior_facet_integral = boost::shared_ptr<ufc::exterior_facet_integral>(form.create_default_exterior_facet_integral());
  for (std::size_t i = 0; i < form.num_exterior_facet_domains(); i++)
    exterior_facet_integrals.push_back(boost::shared_ptr<ufc::exterior_facet_integral>(form.create_exterior_facet_integral(i)));

  // Create interior facet integrals
  default_interior_facet_integral = boost::shared_ptr<ufc::interior_facet_integral>(form.create_default_interior_facet_integral());
  for (std::size_t i = 0; i < form.num_interior_facet_domains(); i++)
    interior_facet_integrals.push_back(boost::shared_ptr<ufc::interior_facet_integral>(form.create_interior_facet_integral(i)));

  // Get maximum local dimensions
  std::vector<std::size_t> max_local_dimension;
  std::vector<std::size_t> max_macro_local_dimension;
  for (std::size_t i = 0; i < form.rank(); i++)
  {
    dolfin_assert(V[i]->dofmap());
    max_local_dimension.push_back(V[i]->dofmap()->max_cell_dimension());
    max_macro_local_dimension.push_back(2*V[i]->dofmap()->max_cell_dimension());
  }

  // Initialize local tensor
  std::size_t num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_local_dimension[i];
  A.resize(num_entries);
  A_facet.resize(num_entries);

  // Initialize local tensor for macro element
  num_entries = 1;
  for (std::size_t i = 0; i < form.rank(); i++)
    num_entries *= max_macro_local_dimension[i];
  macro_A.resize(num_entries);

  // Initialize coefficients
  _w.resize(form.num_coefficients());
  w_pointer.resize(form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    _w[i].resize(coefficient_elements[i].space_dimension());
    w_pointer[i] = &_w[i][0];
  }

  // Initialize coefficients on macro element
  _macro_w.resize(form.num_coefficients());
  macro_w_pointer.resize(form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    const std::size_t n = 2*coefficient_elements[i].space_dimension();
    _macro_w[i].resize(n);
    macro_w_pointer[i] = &_macro_w[i][0];
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c)
{
  // Update UFC cell
  cell.update(c);

  // Restrict coefficients to cell
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(&_w[i][0], coefficient_elements[i], c, cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c, std::size_t local_facet)
{
  // Update UFC cell
  cell.update(c, local_facet);

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(&_w[i][0], coefficient_elements[i], c, cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& c0, std::size_t local_facet0,
                 const Cell& c1, std::size_t local_facet1)
{
  // Update UFC cells
  cell0.update(c0, local_facet0);
  cell1.update(c1, local_facet1);

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    const std::size_t offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(&_macro_w[i][0], coefficient_elements[i],
                              c0, cell0);
    coefficients[i]->restrict(&_macro_w[i][0] + offset, coefficient_elements[i],
                              c1, cell1);
  }
}
//-----------------------------------------------------------------------------
