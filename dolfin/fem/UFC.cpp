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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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
UFC::UFC(const Form& form)
 : form(form.ufc_form()), cell(form.mesh()), cell0(form.mesh()),
   cell1(form.mesh()), coefficients(form.coefficients()), dolfin_form(form)
{
  init(form);
}
//-----------------------------------------------------------------------------
UFC::UFC(const UFC& ufc) : form(ufc.dolfin_form.ufc_form()),
   cell(ufc.dolfin_form.mesh()),  cell0(ufc.dolfin_form.mesh()),
   cell1(ufc.dolfin_form.mesh()), coefficients(ufc.dolfin_form.coefficients()),
   dolfin_form(ufc.dolfin_form)
{
  this->init(ufc.dolfin_form);
}
//-----------------------------------------------------------------------------
UFC::~UFC()
{
  //const uint rank = this->form.rank();
  const uint num_coefficients = this->form.num_coefficients();

  // Delete coefficients
  for (uint i = 0; i < num_coefficients; i++)
    delete [] w[i];
  delete [] w;

  // Delete macro coefficients
  for (uint i = 0; i < num_coefficients; i++)
    delete [] macro_w[i];
  delete [] macro_w;
}
//-----------------------------------------------------------------------------
void UFC::init(const Form& form)
{
  // Initialize mesh
  this->mesh.init(form.mesh());

  // Get function spaces for arguments
  std::vector<boost::shared_ptr<const FunctionSpace> > V = form.function_spaces();

  // Create finite elements for coefficients
  for (uint i = 0; i < this->form.num_coefficients(); i++)
  {
    boost::shared_ptr<ufc::finite_element> element(this->form.create_finite_element(this->form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }

  // Create cell integrals
  for (uint i = 0; i < this->form.num_cell_domains(); i++)
    cell_integrals.push_back(boost::shared_ptr<ufc::cell_integral>(this->form.create_cell_integral(i)));

  // Create exterior facet integrals
  for (uint i = 0; i < this->form.num_exterior_facet_domains(); i++)
    exterior_facet_integrals.push_back(boost::shared_ptr<ufc::exterior_facet_integral>(this->form.create_exterior_facet_integral(i)));

  // Create interior facet integrals
  for (uint i = 0; i < this->form.num_interior_facet_domains(); i++)
    interior_facet_integrals.push_back(boost::shared_ptr<ufc::interior_facet_integral>(this->form.create_interior_facet_integral(i)));

  // Get maximum local dimensions
  std::vector<uint> max_local_dimension;
  std::vector<uint> max_macro_local_dimension;
  for (uint i = 0; i < this->form.rank(); i++)
  {
    max_local_dimension.push_back(V[i]->dofmap().max_cell_dimension());
    max_macro_local_dimension.push_back(2*V[i]->dofmap().max_cell_dimension());
  }

  // Initialize local tensor
  uint num_entries = 1;
  for (uint i = 0; i < this->form.rank(); i++)
    num_entries *= max_local_dimension[i];
  A.reset(new double[num_entries]);
  A_facet.reset(new double[num_entries]);

  // Initialize local tensor for macro element
  num_entries = 1;
  for (uint i = 0; i < this->form.rank(); i++)
    num_entries *= max_macro_local_dimension[i];
  macro_A.reset(new double[num_entries]);

  // Initialize coefficients
  w = new double*[this->form.num_coefficients()];
  for (uint i = 0; i < this->form.num_coefficients(); i++)
    w[i] = new double[coefficient_elements[i].space_dimension()];

  // Initialize coefficients on macro element
  macro_w = new double*[this->form.num_coefficients()];
  for (uint i = 0; i < this->form.num_coefficients(); i++)
  {
    const uint n = 2*coefficient_elements[i].space_dimension();
    macro_w[i] = new double[n];
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell)
{
  // Update UFC cell
  this->cell.update(cell);

  // Restrict coefficients to cell
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(w[i], coefficient_elements[i], cell, this->cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell, uint local_facet)
{
  // Update UFC cell
  this->cell.update(cell, local_facet);

  // Restrict coefficients to facet
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->restrict(w[i], coefficient_elements[i], cell, this->cell);
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell0, uint local_facet0,
                 const Cell& cell1, uint local_facet1)
{
  // Update UFC cells
  this->cell0.update(cell0, local_facet0);
  this->cell1.update(cell1, local_facet1);

  // Restrict coefficients to facet
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    const uint offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(macro_w[i], coefficient_elements[i],
                              cell0, this->cell0);
    coefficients[i]->restrict(macro_w[i] + offset, coefficient_elements[i],
                              cell1, this->cell1);
  }
}
//-----------------------------------------------------------------------------
