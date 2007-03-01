// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/UFCData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UFCData::UFCData(const ufc::form& form) : form(form)
{
  num_arguments = form.rank() + form.num_coefficients();

  // Create finite elements
  finite_elements = new ufc::finite_element* [num_arguments];
  for (uint i = 0; i < num_arguments; i++)
    finite_elements[i] = form.create_finite_element(i);

  // Create dof maps
  dof_maps = new ufc::dof_map* [num_arguments];
  for (uint i = 0; i < num_arguments; i++)
    dof_maps[i] = form.create_dof_map(i);

  // FIXME: Assume for now there is only one sub domain

  // Create integrals
  cell_integral = form.create_cell_integral(0);
  exterior_facet_integral = form.create_exterior_facet_integral(0);
  interior_facet_integral = form.create_interior_facet_integral(0);
}
//-----------------------------------------------------------------------------
UFCData::~UFCData()
{
  // Delete finite elements
  for (uint i = 0; i < num_arguments; i++)
    delete finite_elements[i];
  delete [] finite_elements;

  // Delete dof maps
  for (uint i = 0; i < num_arguments; i++)
    delete dof_maps[i];
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
