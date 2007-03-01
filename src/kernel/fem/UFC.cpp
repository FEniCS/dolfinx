// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

#include <dolfin/constants.h>
#include <dolfin/DofMaps.h>
#include <dolfin/DofMap.h>
#include <dolfin/UFC.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UFC::UFC(const ufc::form& form, DofMaps& dof_maps) : form(form)
{
  // Compute the number of arguments
  num_arguments = form.rank() + form.num_coefficients();

  // Create finite elements
  finite_elements = new ufc::finite_element*[num_arguments];
  for (uint i = 0; i < num_arguments; i++)
    finite_elements[i] = form.create_finite_element(i);

  // Create dof maps (reuse from dof map storage)
  this->dof_maps = new ufc::dof_map*[num_arguments];
  for (uint i = 0; i < num_arguments; i++)
    this->dof_maps[i] = &dof_maps[i].ufc_dof_map;

  // FIXME: Assume for now there is only one sub domain

  // Create integrals
  cell_integral = form.create_cell_integral(0);
  exterior_facet_integral = form.create_exterior_facet_integral(0);
  interior_facet_integral = form.create_interior_facet_integral(0);
}
//-----------------------------------------------------------------------------
UFC::~UFC()
{
  // Delete finite elements
  for (uint i = 0; i < num_arguments; i++)
    delete finite_elements[i];
  delete [] finite_elements;

  // Delete dof maps (don't touch reused dof maps)
  delete [] dof_maps;

  // Delete integrals
  if ( cell_integral )
    delete cell_integral;
  if ( exterior_facet_integral )
    delete exterior_facet_integral;
  if ( interior_facet_integral )
    delete interior_facet_integral;
}
//-----------------------------------------------------------------------------
