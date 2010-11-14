// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2009
// Modified by Garth N. Wells, 2009
//
// First added:  2007-01-17
// Last changed: 2010-11-09

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
  // Delete dofs
  for (uint i = 0; i < this->form.rank(); i++)
    delete [] dofs[i];
  delete [] dofs;

  // Delete macro dofs
  for (uint i = 0; i < this->form.rank(); i++)
    delete [] macro_dofs[i];
  delete [] macro_dofs;

  // Delete coefficients
  for (uint i = 0; i < this->form.num_coefficients(); i++)
    delete [] w[i];
  delete [] w;

  // Delete macro coefficients
  for (uint i = 0; i < this->form.num_coefficients(); i++)
    delete [] macro_w[i];
  delete [] macro_w;
}
//-----------------------------------------------------------------------------
void UFC::init(const Form& form)
{
  // Create finite elements
  for (uint i = 0; i < this->form.rank(); i++)
  {
    boost::shared_ptr<ufc::finite_element> element(this->form.create_finite_element(i));
    finite_elements.push_back( FiniteElement(element) );
  }

  // Create finite elements for coefficients
  for (uint i = 0; i < this->form.num_coefficients(); i++)
  {
    boost::shared_ptr<ufc::finite_element> element(this->form.create_finite_element(this->form.rank() + i));
    coefficient_elements.push_back( FiniteElement(element) );
  }

  // Create cell integrals
  for (uint i = 0; i < this->form.num_cell_integrals(); i++)
    cell_integrals.push_back( boost::shared_ptr<ufc::cell_integral>(this->form.create_cell_integral(i)) );

  // Create exterior facet integrals
  for (uint i = 0; i < this->form.num_exterior_facet_integrals(); i++)
    exterior_facet_integrals.push_back( boost::shared_ptr<ufc::exterior_facet_integral>(this->form.create_exterior_facet_integral(i)) );

  // Create interior facet integrals
  for (uint i = 0; i < this->form.num_interior_facet_integrals(); i++)
    interior_facet_integrals.push_back( boost::shared_ptr<ufc::interior_facet_integral>(this->form.create_interior_facet_integral(i)) );

  // Initialize mesh
  this->mesh.init(form.mesh());

  // Get function spaces for arguments
  std::vector<boost::shared_ptr<const FunctionSpace> > V = form.function_spaces();

  // Initialize local tensor
  uint num_entries = 1;
  for (uint i = 0; i < this->form.rank(); i++)
    num_entries *= V[i]->dofmap().max_local_dimension();
  A.reset(new double[num_entries]);
  for (uint i = 0; i < num_entries; i++)
    A[i] = 0.0;

  // Initialize local tensor for macro element
  num_entries = 1;
  for (uint i = 0; i < this->form.rank(); i++)
    num_entries *= 2*V[i]->dofmap().max_local_dimension();
  macro_A.reset(new double[num_entries]);
  for (uint i = 0; i < num_entries; i++)
    macro_A[i] = 0.0;

  // Allocate memory for storing local dimensions
  local_dimensions.reset(new uint[this->form.rank()]);
  macro_local_dimensions.reset(new uint[this->form.rank()]);

  // Initialize global dimensions
  global_dimensions.reset(new uint[this->form.rank()]);
  for (uint i = 0; i < this->form.rank(); i++)
    global_dimensions[i] = V[i]->dofmap().global_dimension();

  // Initialize dofs
  dofs = new uint*[this->form.rank()];
  for (uint i = 0; i < this->form.rank(); i++)
  {
    dofs[i] = new uint[V[i]->dofmap().max_local_dimension()];
    for (uint j = 0; j < V[i]->dofmap().max_local_dimension(); j++)
      dofs[i][j] = 0;
  }
  //dofs.ressize(this->form.rank());
  //for (uint i = 0; i < this->form.rank(); i++)
  //  dof[i].resize(local_dimensions[i]);

  // Initialize dofs on macro element
  macro_dofs = new uint*[this->form.rank()];
  for (uint i = 0; i < this->form.rank(); i++)
  {
    const uint max_dimension = 2*V[i]->dofmap().max_local_dimension();
    macro_dofs[i] = new uint[max_dimension];
    for (uint j = 0; j < max_dimension; j++)
      macro_dofs[i][j] = 0;
  }

  // Initialize coefficients
  w = new double*[this->form.num_coefficients()];
  for (uint i = 0; i < this->form.num_coefficients(); i++)
  {
    const uint n = coefficient_elements[i].space_dimension();
    w[i] = new double[n];
    for (uint j = 0; j < n; j++)
      w[i][j] = 0.0;
  }

  // Initialize coefficients on macro element
  macro_w = new double*[this->form.num_coefficients()];
  for (uint i = 0; i < this->form.num_coefficients(); i++)
  {
    const uint n = 2*coefficient_elements[i].space_dimension();
    macro_w[i] = new double[n];
    for (uint j = 0; j < n; j++)
      macro_w[i][j] = 0.0;
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell)
{
  // Update UFC cell
  this->cell.update(cell);

  // Update local dimensions
  for (uint i = 0; i < form.rank(); i++)
    local_dimensions[i] = dolfin_form.function_space(i)->dofmap().local_dimension(this->cell);

  // Restrict coefficients to cell
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    coefficients[i]->restrict(w[i], coefficient_elements[i], cell,
                              this->cell, -1);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell, uint local_facet)
{
  // Update UFC cell
  this->cell.update(cell);

  // Update local dimensions
  for (uint i = 0; i < form.rank(); i++)
    local_dimensions[i] = dolfin_form.function_space(i)->dofmap().local_dimension(this->cell);

  // Restrict coefficients to facet
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    coefficients[i]->restrict(w[i], coefficient_elements[i], cell, this->cell,
                              local_facet);
  }
}
//-----------------------------------------------------------------------------
void UFC::update(const Cell& cell0, uint local_facet0,
                 const Cell& cell1, uint local_facet1)
{
  // Update UFC cells
  this->cell0.update(cell0);
  this->cell1.update(cell1);

  // Update local dimensions
  for (uint i = 0; i < form.rank(); i++)
  {
    macro_local_dimensions[i]
      = dolfin_form.function_space(i)->dofmap().local_dimension(this->cell0)
      + dolfin_form.function_space(i)->dofmap().local_dimension(this->cell1);
  }

  // Restrict coefficients to facet
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    const uint offset = coefficient_elements[i].space_dimension();
    coefficients[i]->restrict(macro_w[i],          coefficient_elements[i], cell0, this->cell0, local_facet0);
    coefficients[i]->restrict(macro_w[i] + offset, coefficient_elements[i], cell1, this->cell1, local_facet1);
  }
}
//-----------------------------------------------------------------------------
