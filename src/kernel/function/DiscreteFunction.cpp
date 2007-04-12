// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-02
// Last changed: 2007-04-12

#include <dolfin/dolfin_log.h>
#include <dolfin/Form.h>
#include <dolfin/DofMap.h>
#include <dolfin/DiscreteFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, const Form& form, uint i)
  : GenericFunction(), finite_element(0), dof_map(0), dofs(0)
{
  // Check argument
  const uint num_arguments = form.form().rank() + form.form().num_coefficients();
  if ( i >= num_arguments )
  {
    dolfin_error2("Illegal function index %d. Form only has %d arguments.",
                  i, num_arguments);
  }

  // Create finite element
  finite_element = form.form().create_finite_element(i);

  // Create dof map
  ufc::dof_map* ufc_dof_map = form.form().create_dof_map(i);
  dof_map = new DofMap(*ufc_dof_map, mesh);
  delete ufc_dof_map;

  // Initialize vector
  x.init(dof_map->global_dimension());

  // Initialize local array for mapping of dofs
  dofs = new uint[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
    dofs[i] = 0;
}
//-----------------------------------------------------------------------------
DiscreteFunction::~DiscreteFunction()
{
  if ( finite_element )
    delete finite_element;
      
  if ( dof_map )
    delete dof_map;

  if ( dofs )
    delete [] dofs;
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::rank() const
{
  dolfin_assert(finite_element);
  return finite_element->value_rank();
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::dim(uint i) const
{
  dolfin_assert(finite_element);
  return finite_element->value_dimension(i);
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* values, Mesh& mesh)
{
  dolfin_assert(values);
  dolfin_assert(finite_element);
  
  // Interpolate vertex values on each cell and pick the last value
  // if two or more cells disagree on the vertex values
  CellIterator cell(mesh);
  UFCCell ufc_cell(*cell);
  for (; !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Interpolate values at the vertices
    //finite_element->interpolate_vertex_values(vertex_values,
    //                                         dof_values,
    //                                         ufc_cell);

    // Copy values to array

    // FIXME: In preparation...
  }
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element)
{
  dolfin_assert(coefficients);
  dolfin_assert(this->finite_element);
  dolfin_assert(this->dof_map);
  dolfin_assert(this->dofs);

  // FIXME: Better test here, compare against the local element

  // Check dimension
  if ( finite_element.space_dimension() != dof_map->local_dimension() )
    dolfin_error("Finite element does not match for interpolation of discrete function.");

  // Tabulate dofs
  dof_map->tabulate_dofs(dofs, cell);
  
  // Pick values from global vector
  x.get(coefficients, dof_map->local_dimension(), dofs);
}
//-----------------------------------------------------------------------------
