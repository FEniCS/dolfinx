// Copyright (C) 2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-05-08
// Last changed: 2009-09-28

#include <dolfin/log/log.h>
#include "Data.h"
#include "Function.h"
#include "UFCFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFCFunction::UFCFunction(const Function& v, Data& data)
  : v(v), data(data)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UFCFunction::~UFCFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void UFCFunction::evaluate(double* values,
                           const double* coordinates,
                           const ufc::cell& cell) const
{
  assert(values);

  // Set coordinates
  data.x = coordinates;

  // If we store the cell here, we also need to store the mesh. The
  // cell data might otherwise be falsely used when interpolating to
  // another mesh.
  //data._ufc_cell = &cell;

  // Call eval for function
  v.eval(values, data);
}
//-----------------------------------------------------------------------------
