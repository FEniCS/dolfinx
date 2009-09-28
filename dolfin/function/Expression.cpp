// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-09-28

#include <dolfin/log/log.h>
#include <dolfin/fem/FiniteElement.h>
#include "FunctionSpace.h"
#include "Data.h"
#include "Expression.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Expression::Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::~Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Expression::eval(double* values, const double* x) const
{
  assert(values);
  assert(x);

  // Missing eval method if we reach this point
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
void Expression::eval(double* values, const Data& data) const
{
  assert(values);
  assert(data.is_valid());

  // Redirect to simple eval
  eval(values, data.x);
}
//-----------------------------------------------------------------------------
void Expression::restrict(double* w,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          const FunctionSpace& V,
                          int local_facet) const
{
  assert(w);

  // Update cell data
  data.update(dolfin_cell, ufc_cell, local_facet);

  // Evaluate each dof to get the expansion coefficients
  const FiniteElement& element = V.element();
  for (uint i = 0; i < element.space_dimension(); ++i)
    w[i] = element.evaluate_dof(i, *this, ufc_cell);

  // Invalidate cell data
  data.invalidate();
}
//-----------------------------------------------------------------------------
void Expression::evaluate(double* values,
                          const double* coordinates,
                          const ufc::cell& cell) const
{
  assert(values);
  assert(coordinates);
  assert(data.is_valid());

  // Update coordinates
  data.x = coordinates;

  // Redirect to eval
  eval(values, data);

  // Reset coordinates
  data.x = 0;
}
//-----------------------------------------------------------------------------
