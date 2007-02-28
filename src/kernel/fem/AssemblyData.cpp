// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#include <dolfin/AssemblyData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AssemblyData::AssemblyData(const ufc::form& form)
{
  // Do nothing

  // FIXME: Assume for now there is only one sub domain

  // Create integrals
  cell_integral = form.create_cell_integral(0);
  exterior_facet_integral = form.create_exterior_facet_integral(0);
  interior_facet_integral = form.create_interior_facet_integral(0);
}
//-----------------------------------------------------------------------------
AssemblyData::~AssemblyData()
{
  // Delete integrals
  if ( cell_integral )
    delete cell_integral;
  if ( exterior_facet_integral )
    delete exterior_facet_integral;
  if ( interior_facet_integral )
    delete interior_facet_integral;
}
//-----------------------------------------------------------------------------
