// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2008-06-09

#include <dolfin/common/types.h>
#include "DofMapSet.h"
#include "DofMap.h"
#include "UFC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const ufc::form& form, Mesh& mesh, const DofMapSet& dof_map_set) : form(form)
{
  // Create finite elements
  finite_elements = new ufc::finite_element*[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
    finite_elements[i] = form.create_finite_element(i);

  // Create finite elements for coefficients
  coefficient_elements = new ufc::finite_element*[form.num_coefficients()];
  for (uint i = 0; i < form.num_coefficients(); i++)
    coefficient_elements[i] = form.create_finite_element(form.rank() + i);

  // Create cell integrals
  cell_integrals = new ufc::cell_integral*[form.num_cell_integrals()];
  for (uint i = 0; i < form.num_cell_integrals(); i++)
    cell_integrals[i] = form.create_cell_integral(i);

  // Create exterior facet integrals
  exterior_facet_integrals = new ufc::exterior_facet_integral*[form.num_exterior_facet_integrals()];
  for (uint i = 0; i < form.num_exterior_facet_integrals(); i++)
    exterior_facet_integrals[i] = form.create_exterior_facet_integral(i);

  // Create interior facet integrals
  interior_facet_integrals = new ufc::interior_facet_integral*[form.num_interior_facet_integrals()];
  for (uint i = 0; i < form.num_interior_facet_integrals(); i++)
    interior_facet_integrals[i] = form.create_interior_facet_integral(i);

  // Initialize mesh
  this->mesh.init(mesh);

  // Initialize cells with first cell in mesh
  CellIterator cell(mesh);
  this->cell.init(*cell);
  this->cell0.init(*cell);
  this->cell1.init(*cell);

  // Initialize local tensor
  num_entries = 1;
  for (uint i = 0; i < form.rank(); i++)
    num_entries *= dof_map_set[i].local_dimension();
  A = new real[num_entries];
  for (uint i = 0; i < num_entries; i++)
    A[i] = 0.0;

  // Initialize local tensor for macro element
  num_entries = 1;
  for (uint i = 0; i < form.rank(); i++)
    num_entries *= 2*dof_map_set[i].local_dimension();
  macro_A = new real[num_entries];
  for (uint i = 0; i < num_entries; i++)
    macro_A[i] = 0.0;  

  // Initialize local dimensions
  local_dimensions = new uint[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
    local_dimensions[i] = dof_map_set[i].local_dimension();

  // Initialize local dimensions for macro element
  macro_local_dimensions = new uint[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
    macro_local_dimensions[i] = 2*dof_map_set[i].local_dimension();

  // Initialize global dimensions
  global_dimensions = new uint[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
    global_dimensions[i] = dof_map_set[i].global_dimension();

  // Initialize dofs
  dofs = new uint*[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
  {
    dofs[i] = new uint[local_dimensions[i]];
    for (uint j = 0; j < local_dimensions[i]; j++)
      dofs[i][j] = 0;
  }

  // Initialize dofs on macro element
  macro_dofs = new uint*[form.rank()];
  for (uint i = 0; i < form.rank(); i++)
  {
    macro_dofs[i] = new uint[macro_local_dimensions[i]];
    for (uint j = 0; j < macro_local_dimensions[i]; j++)
      macro_dofs[i][j] = 0;
  }

  // Initialize coefficients
  w = new real*[form.num_coefficients()];
  for (uint i = 0; i < form.num_coefficients(); i++)
  {
    const uint n = coefficient_elements[i]->space_dimension();
    w[i] = new real[n];
    for (uint j = 0; j < n; j++)
      w[i][j] = 0.0;
  }

  // Initialize coefficients on macro element
  macro_w = new real*[form.num_coefficients()];
  for (uint i = 0; i < form.num_coefficients(); i++)
  {
    const uint n = 2*coefficient_elements[i]->space_dimension();
    macro_w[i] = new real[n];
    for (uint j = 0; j < n; j++)
      macro_w[i][j] = 0.0;
  }
}
//-----------------------------------------------------------------------------
UFC::~UFC()
{
  // Delete finite elements
  for (uint i = 0; i < form.rank(); i++)
    delete finite_elements[i];
  delete [] finite_elements;

  // Delete coefficient finite elements
  for (uint i = 0; i < form.num_coefficients(); i++)
    delete coefficient_elements[i];
  delete [] coefficient_elements;

  // Delete cell integrals
  for (uint i = 0; i < form.num_cell_integrals(); i++)
    delete cell_integrals[i];
  delete [] cell_integrals;

  // Delete exterior facet integrals
  for (uint i = 0; i < form.num_exterior_facet_integrals(); i++)
    delete exterior_facet_integrals[i];
  delete [] exterior_facet_integrals;

  // Delete interior facet integrals
  for (uint i = 0; i < form.num_interior_facet_integrals(); i++)
    delete interior_facet_integrals[i];
  delete [] interior_facet_integrals;

  // Delete local tensor
  delete [] A;

  // Delete local tensor for macro element
  delete [] macro_A;

  // Delete local dimensions
  delete [] local_dimensions;

  // Delete global dimensions
  delete [] global_dimensions;

  // Delete local dimensions for macro element
  delete [] macro_local_dimensions;

  // Delete dofs
  for (uint i = 0; i < form.rank(); i++)
    delete [] dofs[i];
  delete [] dofs;

  // Delete macro dofs
  for (uint i = 0; i < form.rank(); i++)
    delete [] macro_dofs[i];
  delete [] macro_dofs;

  // Delete coefficients
  for (uint i = 0; i < form.num_coefficients(); i++)
    delete [] w[i];
  delete [] w;

  // Delete macro coefficients
  for (uint i = 0; i < form.num_coefficients(); i++)
    delete [] macro_w[i];
  delete [] macro_w;
}
//-----------------------------------------------------------------------------
void UFC::update(Cell& cell)
{
  // Update UFC cell
  this->cell.update(cell);

  // FIXME: Update coefficients
}
//-----------------------------------------------------------------------------
void UFC::update(Cell& cell0, Cell& cell1)
{
  // Update UFC cells
  this->cell0.update(cell0);
  this->cell1.update(cell1);

  // FIXME: Update coefficients
}
//-----------------------------------------------------------------------------
