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
UFC::UFC(const Form& form) :
  form(*form.ufc_form()), cell(form.mesh()),
  cell0(form.mesh()), cell1(form.mesh()),
  coefficients(form.coefficients()), dolfin_form(form)
{
  dolfin_assert(form.ufc_form());
  init(form);
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
void UFC::init(const Form& form)
{
  // Initialize mesh dimensions // TODO MSA: Do we need this?
  //this->num_global_mesh_entities(form.mesh().topology().dim() + 1);
  //for (std::size_t d = 0; d < this->num_global_mesh_entities.size(); d++)
  //  this->num_global_mesh_entities[d] = form.mesh().size_global(d);

  // Get function spaces for arguments
  std::vector<boost::shared_ptr<const FunctionSpace> > V = form.function_spaces();

  // Create finite elements for coefficients
  for (std::size_t i = 0; i < this->form.num_coefficients(); i++)
  {
    boost::shared_ptr<ufc::finite_element> element(this->form.create_finite_element(this->form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Create cell integrals
  for (std::size_t i = 0; i < this->form.num_cell_domains(); i++)
    cell_integrals.push_back(boost::shared_ptr<ufc::cell_integral>(this->form.create_cell_integral(i)));

  // Create exterior facet integrals
  for (std::size_t i = 0; i < this->form.num_exterior_facet_domains(); i++)
    exterior_facet_integrals.push_back(boost::shared_ptr<ufc::exterior_facet_integral>(this->form.create_exterior_facet_integral(i)));

  // Create interior facet integrals
  for (std::size_t i = 0; i < this->form.num_interior_facet_domains(); i++)
    interior_facet_integrals.push_back(boost::shared_ptr<ufc::interior_facet_integral>(this->form.create_interior_facet_integral(i)));

  // Get maximum local dimensions
  std::vector<std::size_t> max_local_dimension;
  std::vector<std::size_t> max_macro_local_dimension;
  for (std::size_t i = 0; i < this->form.rank(); i++)
  {
    dolfin_assert(V[i]->dofmap());
    max_local_dimension.push_back(V[i]->dofmap()->max_cell_dimension());
    max_macro_local_dimension.push_back(2*V[i]->dofmap()->max_cell_dimension());
  }

  // Initialize local tensor
  std::size_t num_entries = 1;
  for (std::size_t i = 0; i < this->form.rank(); i++)
    num_entries *= max_local_dimension[i];
  A.resize(num_entries);
  A_facet.resize(num_entries);

  // Initialize local tensor for macro element
  num_entries = 1;
  for (std::size_t i = 0; i < this->form.rank(); i++)
    num_entries *= max_macro_local_dimension[i];
  macro_A.resize(num_entries);

  // Initialize coefficients
  _w.resize(this->form.num_coefficients());
  w_pointer.resize(this->form.num_coefficients());
  for (std::size_t i = 0; i < this->form.num_coefficients(); i++)
  {
    _w[i].resize(coefficient_elements[i].space_dimension());
    w_pointer[i] = &_w[i][0];
  }

  // Initialize coefficients on macro element
  _macro_w.resize(this->form.num_coefficients());
  macro_w_pointer.resize(this->form.num_coefficients());
  for (std::size_t i = 0; i < this->form.num_coefficients(); i++)
  {
    const std::size_t n = 2*coefficient_elements[i].space_dimension();
    _macro_w[i].resize(n);
    macro_w_pointer[i] = &_macro_w[i][0];
  }

  // Initialize vertex coordinates
  const CellType& cell_type = form.mesh().type();
  const std::size_t n = cell_type.num_entities(0)*form.mesh().geometry().dim();
  vertex_coordinates.resize(n);
  vertex_coordinates_0.resize(n);
  vertex_coordinates_1.resize(n);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell)
{
  // Update UFC cell
  this->cell.update(cell);

  // Restrict coefficients to cell
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(&_w[i][0], coefficient_elements[i], cell, this->cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell, std::size_t local_facet)
{
  // Update UFC cell
  this->cell.update(cell, local_facet);

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(&_w[i][0], coefficient_elements[i], cell, this->cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell0, std::size_t local_facet0,
                 const Cell& cell1, std::size_t local_facet1)
{
  // Update UFC cells
  this->cell0.update(cell0, local_facet0);
  this->cell1.update(cell1, local_facet1);

  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    const std::size_t offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(&_macro_w[i][0], coefficient_elements[i],
                              cell0, this->cell0);
    coefficients[i]->restrict(&_macro_w[i][0] + offset, coefficient_elements[i],
                              cell1, this->cell1);
  }
}
//-----------------------------------------------------------------------------
