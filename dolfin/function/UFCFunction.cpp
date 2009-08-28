// Copyright (C) 2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-05-08
// Last changed: 2009-08-29

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

  // Set coordinates and UFC cell
  data.x = coordinates;
  data._ufc_cell = &cell;

  // Call eval for function
  v.eval(values, data);

  // Invalidate eval data
  data.invalidate();
}
//-----------------------------------------------------------------------------
