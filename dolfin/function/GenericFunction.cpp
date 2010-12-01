// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010.
//
// First added:  2009-09-28
// Last changed: 2010-11-30

#include <dolfin/fem/FiniteElement.h>
#include "Data.h"
#include "GenericFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction() : Variable("u", "a function")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunction::~GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint GenericFunction::value_size() const
{
  uint size = 1;
  for (uint i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
void GenericFunction::evaluate(double* values,
                               const double* coordinates,
                               const ufc::cell& cell) const
{
  assert(values);
  assert(coordinates);

  // Add ufc::cell and coordinates to data
  // FIXME: Can creation of Data objects be made more efficient?
  Data data;
  data.set(cell, coordinates);

  Array<double> _values(value_size(), values);

  // Redirect to eval
  eval(_values, data);
}
//-----------------------------------------------------------------------------
void GenericFunction::restrict_as_ufc_function(double* w,
                                               const FiniteElement& element,
                                               const Cell& dolfin_cell,
                                               const ufc::cell& ufc_cell,
                                               int local_facet) const
{
  assert(w);

  // Evaluate each dof to get the expansion coefficients
  for (uint i = 0; i < element.space_dimension(); ++i)
    w[i] = element.evaluate_dof(i, *this, ufc_cell);
}
//-----------------------------------------------------------------------------
